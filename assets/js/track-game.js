/* Drive a circuit from the cockpit, on the real track outline. Arcade handling, flat ground.
   The track is real car position data. Water, buildings and trees come from OpenStreetMap; barriers,
   bridges and grandstands are generic.
   Lap times compare with your own laps, not real ones. */
(function (root) {
  'use strict';

  var THREE_URL = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  // Lap length in metres sets the scale, since the stored outline has none.
  var LENGTHS = { albert_park: 5278 };
  var HALF_WIDTH = 8;
  var SPACING = 4;
  var TOP_SPEED = 92;
  var OFF_SPEED = 32;

  function parsePath(d) {
    var nums = d.match(/-?\d+(\.\d+)?/g).map(Number);
    var pts = [];
    for (var i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
    return pts;
  }

  // Close the lap and resample it evenly, with a light smoothing of the position noise.
  function buildTrack(d, length) {
    var raw = parsePath(d);
    var i, n = raw.length, len = 0;
    for (i = 0; i < n; i++) {
      var a = raw[i], b = raw[(i + 1) % n];
      len += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    var scale = length / len;
    var count = Math.round(length / SPACING);
    var pts = [], seg = 0, into = 0;
    var segLen = function (k) { var p = raw[k], q = raw[(k + 1) % n]; return Math.hypot(q[0] - p[0], q[1] - p[1]) * scale; };
    for (i = 0; i < count; i++) {
      while (into > segLen(seg) && seg < n - 1) { into -= segLen(seg); seg++; }
      var p = raw[seg], q = raw[(seg + 1) % n], t = segLen(seg) ? Math.min(1, into / segLen(seg)) : 0;
      pts.push([(p[0] + (q[0] - p[0]) * t) * scale, (p[1] + (q[1] - p[1]) * t) * scale]);
      into += length / count;
    }
    for (var pass = 0; pass < 4; pass++) {
      pts = pts.map(function (_, k) {
        var s = [0, 0];
        for (var o = -4; o <= 4; o++) { var v = pts[(k + o + count) % count]; s[0] += v[0]; s[1] += v[1]; }
        return [s[0] / 9, s[1] / 9];
      });
    }
    return pts;
  }

  // The driving model, free of the page so it can be stepped on its own.
  function createSim(d, length) {
    var pts = buildTrack(d, length || 5000), N = pts.length;
    var sim = {
      pts: pts, x: 0, y: 0, heading: 0, speed: 0, steer: 0, idx: 0, progress: 0, offTrack: false, wrongWay: false,
      running: false, lapTime: 0, offTime: 0, lastLap: null, bestLap: null, lapValid: true, laps: 0, onLap: null
    };
    function tangent(i) {
      var a = pts[(i - 2 + N) % N], b = pts[(i + 2) % N];
      return Math.atan2(b[1] - a[1], b[0] - a[0]);
    }
    sim.tangent = tangent;
    function nearest(from, span) {
      var best = from, bd = Infinity;
      for (var o = -span; o <= span; o++) {
        var k = (from + o + N) % N, dx = pts[k][0] - sim.x, dy = pts[k][1] - sim.y, q = dx * dx + dy * dy;
        if (q < bd) { bd = q; best = k; }
      }
      return { i: best, dist: Math.sqrt(bd) };
    }
    sim.reset = function (i) {
      sim.idx = i || 0;
      sim.x = pts[sim.idx][0]; sim.y = pts[sim.idx][1];
      sim.heading = tangent(sim.idx);
      sim.speed = 0; sim.steer = 0; sim.offTrack = false; sim.wrongWay = false;
    };
    sim.restart = function () {
      sim.reset(0);
      sim.progress = 0; sim.running = false; sim.lapTime = 0; sim.offTime = 0; sim.lapValid = true;
    };
    sim.step = function (dt, input) {
      if (!sim.running && input.up) sim.running = true;
      if (sim.running) sim.lapTime += dt;

      var limit = sim.offTrack ? OFF_SPEED : TOP_SPEED;
      if (input.up) sim.speed += (sim.speed < limit ? 15 * (1 - sim.speed / limit * 0.75) : 0) * dt;
      if (input.down) sim.speed -= (sim.speed > 0 ? 42 : 8) * dt;
      sim.speed -= sim.speed * (input.up ? 0.02 : 0.12) * dt;
      if (sim.speed > limit) sim.speed = Math.max(limit, sim.speed - 45 * dt); // the grass scrubs speed off
      if (sim.speed < -12) sim.speed = -12;
      if (!input.up && !input.down && Math.abs(sim.speed) < 0.3) sim.speed = 0;

      // Keys are on or off; easing the wheel keeps the view from snapping.
      var target = (input.right ? 1 : 0) - (input.left ? 1 : 0);
      var ease = Math.min(1, dt * (target ? 5 : 9));
      sim.steer += (target - sim.steer) * ease;
      var grip = sim.offTrack ? 12 : 40;
      var abs = Math.abs(sim.speed);
      var rate = Math.min(1.9, grip / Math.max(abs, 1)) * Math.min(1, abs / 4);
      sim.heading += sim.steer * rate * (sim.speed < 0 ? -1 : 1) * dt;
      sim.x += Math.cos(sim.heading) * sim.speed * dt;
      sim.y += Math.sin(sim.heading) * sim.speed * dt;

      var at = nearest(sim.idx, 60);
      var delta = at.i - sim.idx;
      if (delta > N / 2) delta -= N;
      if (delta < -N / 2) delta += N;
      sim.idx = at.i;
      sim.progress += delta;
      sim.offTrack = at.dist > HALF_WIDTH + 1;
      if (sim.offTrack && sim.running) sim.offTime += dt;
      if (at.dist > 60) sim.reset(at.i); // through the barriers: back on the line, without the speed

      var diff = Math.atan2(Math.sin(sim.heading - tangent(sim.idx)), Math.cos(sim.heading - tangent(sim.idx)));
      sim.wrongWay = sim.speed > 3 && Math.abs(diff) > Math.PI * 0.6;

      if (sim.progress >= N) {
        var valid = sim.offTime <= 6;
        sim.lastLap = sim.lapTime;
        sim.lapValid = valid;
        sim.laps++;
        if (valid && (sim.bestLap === null || sim.lapTime < sim.bestLap)) sim.bestLap = sim.lapTime;
        if (sim.onLap) sim.onLap(sim.lapTime, valid);
        sim.progress -= N; sim.lapTime = 0; sim.offTime = 0;
      }
    };
    sim.reset(0);
    return sim;
  }

  function fmt(t) {
    if (t === null || t === undefined) return '–';
    var m = Math.floor(t / 60), s = t - m * 60;
    return m + ':' + (s < 10 ? '0' : '') + s.toFixed(3);
  }

  // ---------- scenery ----------

  function canvasTexture(THREE, w, h, paint, repeat) {
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    paint(c.getContext('2d'), w, h);
    var t = new THREE.CanvasTexture(c);
    if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
    t.anisotropy = 8;
    return t;
  }

  function speckle(g, w, h, base, range, count) {
    g.fillStyle = base; g.fillRect(0, 0, w, h);
    for (var i = 0; i < count; i++) {
      var v = Math.floor(Math.random() * range);
      g.fillStyle = 'rgba(' + v + ',' + v + ',' + v + ',.16)';
      g.fillRect(Math.random() * w, Math.random() * h, 2, 2);
    }
  }

  // A strip between two sideways offsets from the centre line, skipping points `keep` rejects.
  function ribbon(THREE, track, off0, off1, y0, y1, material, tile, keep) {
    var pos = [], uv = [], index = [], N = track.pts.length, v = 0, used = 0, prev = false;
    for (var i = 0; i <= N; i++) {
      var k = i % N, p = track.pts[k], nx = track.normal[k][0], nz = track.normal[k][1];
      var on = !keep || keep(k);
      if (on) {
        pos.push(p[0] + nx * off0, y0, p[1] + nz * off0, p[0] + nx * off1, y1, p[1] + nz * off1);
        uv.push(0, v / tile, 1, v / tile);
        if (prev) index.push(used * 2 - 2, used * 2 - 1, used * 2, used * 2 - 1, used * 2 + 1, used * 2);
        used++;
      }
      prev = on;
      v += SPACING;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.computeVertexNormals();
    return new THREE.Mesh(geo, material);
  }

  function analyseTrack(pts) {
    var N = pts.length, normal = [], curve = [], i, grid = {}, CELL = 60;
    var ang = function (k) {
      var a = pts[(k - 3 + N) % N], b = pts[(k + 3) % N];
      return Math.atan2(b[1] - a[1], b[0] - a[0]);
    };
    for (i = 0; i < N; i++) {
      var t = ang(i);
      normal.push([-Math.sin(t), Math.cos(t)]);
      var d = ang((i + 5) % N) - ang((i - 5 + N) % N);
      curve.push(Math.atan2(Math.sin(d), Math.cos(d)) / (10 * SPACING));
      var key = Math.floor(pts[i][0] / CELL) + ':' + Math.floor(pts[i][1] / CELL);
      (grid[key] = grid[key] || []).push(i);
    }
    // Distance to the nearest stretch of track, ignoring the stretch at `skip` if given.
    var clearance = function (x, z, skip) {
      var best = Infinity, cx = Math.floor(x / CELL), cz = Math.floor(z / CELL);
      for (var a = -2; a <= 2; a++) {
        for (var b = -2; b <= 2; b++) {
          var list = grid[(cx + a) + ':' + (cz + b)];
          if (!list) continue;
          for (var j = 0; j < list.length; j++) {
            var q = list[j];
            if (skip !== undefined) {
              var gap = Math.abs(q - skip);
              if (Math.min(gap, N - gap) < 30) continue;
            }
            best = Math.min(best, Math.hypot(pts[q][0] - x, pts[q][1] - z));
          }
        }
      }
      return best;
    };
    return { pts: pts, normal: normal, curve: curve, clearance: clearance };
  }

  function bannerTexture(THREE, text) {
    return canvasTexture(THREE, 1024, 128, function (g, w, h) {
      g.fillStyle = '#17623f'; g.fillRect(0, 0, w, h);
      g.fillStyle = '#f4f4f0';
      g.font = '700 56px "Inter", Arial, sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(text, w / 4, h / 2); g.fillText(text, w * 3 / 4, h / 2);
    });
  }

  function buildWorld(THREE, scene, track, name, real) {
    var N = track.pts.length;
    var lit = function (opts) { return new THREE.MeshLambertMaterial(opts); };

    // Sky
    var skyTex = canvasTexture(THREE, 16, 512, function (g, w, h) {
      var s = g.createLinearGradient(0, 0, 0, h);
      s.addColorStop(0, '#3f7fc4'); s.addColorStop(.45, '#a7cdec'); s.addColorStop(.52, '#e4eef2'); s.addColorStop(1, '#e4eef2');
      g.fillStyle = s; g.fillRect(0, 0, w, h);
    });
    var sky = new THREE.Mesh(new THREE.SphereGeometry(3200, 24, 16), new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, fog: false, depthWrite: false }));
    scene.add(sky);
    scene.userData.sky = sky;

    // Ground, centred on the track
    var minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    track.pts.forEach(function (p) { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); minZ = Math.min(minZ, p[1]); maxZ = Math.max(maxZ, p[1]); });
    var grass = canvasTexture(THREE, 256, 256, function (g, w, h) {
      speckle(g, w, h, '#4d8a3f', 60, 5000);
      g.fillStyle = 'rgba(30,70,25,.12)'; g.fillRect(0, 0, w, h / 2);
    }, true);
    grass.repeat.set(220, 220);
    var ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), lit({ map: grass }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.set((minX + maxX) / 2, -0.05, (minZ + maxZ) / 2);
    scene.add(ground);

    var overlap = function (off) {
      return function (k) {
        var p = track.pts[k], n = track.normal[k];
        return track.clearance(p[0] + n[0] * off, p[1] + n[1] * off, k) > Math.abs(off) + 3;
      };
    };

    // Run-off, tarmac, lines
    var asphaltTex = canvasTexture(THREE, 256, 256, function (g, w, h) { speckle(g, w, h, '#3b3d42', 140, 9000); }, true);
    var runoffTex = canvasTexture(THREE, 256, 256, function (g, w, h) { speckle(g, w, h, '#8d8f8f', 70, 6000); }, true);
    var corner = function (k) {
      for (var o = -14; o <= 14; o += 2) if (Math.abs(track.curve[(k + o + N) % N]) > 0.012) return true;
      return false;
    };
    scene.add(ribbon(THREE, track, -HALF_WIDTH - 8, HALF_WIDTH + 8, 0, 0, lit({ map: runoffTex, side: THREE.DoubleSide }), 30, corner));
    scene.add(ribbon(THREE, track, -HALF_WIDTH, HALF_WIDTH, 0.02, 0.02, lit({ map: asphaltTex, side: THREE.DoubleSide }), 16));
    var white = lit({ color: 0xf0f0ec, side: THREE.DoubleSide });
    scene.add(ribbon(THREE, track, -HALF_WIDTH, -HALF_WIDTH + 0.25, 0.03, 0.03, white, 10));
    scene.add(ribbon(THREE, track, HALF_WIDTH - 0.25, HALF_WIDTH, 0.03, 0.03, white, 10));

    // Kerbs on the corners
    var kerbTex = canvasTexture(THREE, 8, 64, function (g, w, h) {
      g.fillStyle = '#d3211f'; g.fillRect(0, 0, w, h / 2);
      g.fillStyle = '#f2f2ee'; g.fillRect(0, h / 2, w, h / 2);
    }, true);
    var kerb = lit({ map: kerbTex, side: THREE.DoubleSide });
    scene.add(ribbon(THREE, track, -HALF_WIDTH - 1.6, -HALF_WIDTH, 0.06, 0.03, kerb, 5, corner));
    scene.add(ribbon(THREE, track, HALF_WIDTH, HALF_WIDTH + 1.6, 0.03, 0.06, kerb, 5, corner));

    // Barriers and catch fencing, away from anywhere the track passes close to itself
    var boardTex = canvasTexture(THREE, 512, 64, function (g, w, h) {
      g.fillStyle = '#1f7a4d'; g.fillRect(0, 0, w, h);
      g.fillStyle = '#d7c94a'; g.fillRect(0, h - 8, w, 8);
      g.fillStyle = '#eef1ea'; g.font = '700 34px Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(name.toUpperCase(), w / 2, h / 2 - 3);
    }, true);
    var board = lit({ map: boardTex, side: THREE.DoubleSide });
    var fenceTex = canvasTexture(THREE, 64, 64, function (g, w, h) {
      g.clearRect(0, 0, w, h);
      g.strokeStyle = 'rgba(70,75,80,.75)'; g.lineWidth = 1.5;
      for (var i = -h; i < w; i += 10) {
        g.beginPath(); g.moveTo(i, 0); g.lineTo(i + h, h); g.moveTo(i + h, 0); g.lineTo(i, h); g.stroke();
      }
    }, true);
    var fence = new THREE.MeshBasicMaterial({ map: fenceTex, side: THREE.DoubleSide, transparent: true, depthWrite: false });
    [-1, 1].forEach(function (side) {
      var off = -side * (HALF_WIDTH + 18);
      var clear = overlap(off);
      var geoWall = function (h0, h1, mat, tile) {
        var pos = [], uv = [], index = [], used = 0, prev = false, v = 0;
        for (var i = 0; i <= N; i += 2) {
          var k = i % N, p = track.pts[k], n = track.normal[k], on = clear(k);
          if (on) {
            var x = p[0] + n[0] * off, z = p[1] + n[1] * off;
            pos.push(x, h0, z, x, h1, z);
            uv.push(side * v / tile, 0, side * v / tile, 1);
            if (prev) index.push(used * 2 - 2, used * 2 - 1, used * 2, used * 2 - 1, used * 2 + 1, used * 2);
            used++;
          }
          prev = on;
          v += SPACING * 2;
        }
        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
        geo.setIndex(index);
        geo.computeVertexNormals();
        scene.add(new THREE.Mesh(geo, mat));
      };
      geoWall(0, 1.3, board, 24);
      geoWall(1.3, 5, fence, 4);
    });

    // Trees, kept clear of the road and the barriers
    var trunk = new THREE.CylinderGeometry(0.35, 0.5, 5, 6);
    var crown = new THREE.IcosahedronGeometry(5.5, 1);
    var spots = real.trees.filter(function (t) { return track.clearance(t[0], t[1]) > HALF_WIDTH + 22; }).map(function (t) {
      return [t[0], t[1], 0.8 + Math.random() * 1.1];
    });
    var trunks = new THREE.InstancedMesh(trunk, lit({ color: 0x5b4632 }), spots.length);
    var crowns = new THREE.InstancedMesh(crown, lit({ color: 0xffffff }), spots.length);
    var m = new THREE.Matrix4(), col = new THREE.Color();
    spots.forEach(function (s, i) {
      m.makeScale(s[2], s[2], s[2]); m.setPosition(s[0], 2.5 * s[2], s[1]); trunks.setMatrixAt(i, m);
      m.makeScale(s[2], s[2] * 1.15, s[2]); m.setPosition(s[0], 9 * s[2], s[1]); crowns.setMatrixAt(i, m);
      col.setHSL(.27 + Math.random() * .07, .42, .24 + Math.random() * .1);
      crowns.setColorAt(i, col);
    });
    scene.add(trunks); scene.add(crowns);

    // Real water and buildings
    var waterMat = new THREE.MeshPhongMaterial({ color: 0x4f86a8, shininess: 90, specular: 0x8899aa });
    real.water.forEach(function (ring) {
      var shape = new THREE.Shape(ring.map(function (p) { return new THREE.Vector2(p[0], p[1]); }));
      var mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), waterMat);
      mesh.material.side = THREE.DoubleSide;
      mesh.rotation.x = Math.PI / 2; // shape y becomes world z
      mesh.position.y = -0.02;
      scene.add(mesh);
    });
    var pos = [], colors = [], shade = new THREE.Color();
    real.buildings.forEach(function (b) {
      var ring = b.p, h = b.h;
      if (track.clearance(ring[0][0], ring[0][1]) < HALF_WIDTH + 20) return;
      shade.setHSL(.08, .08, .55 + Math.random() * .25);
      var quad = function (a, b_, c_, f) {
        [a, b_, c_].forEach(function (v) { pos.push(v[0], v[1], v[2]); colors.push(shade.r * f, shade.g * f, shade.b * f); });
      };
      ring.forEach(function (p, k) {
        var q = ring[(k + 1) % ring.length];
        var A = [p[0], 0, p[1]], B = [q[0], 0, q[1]], C = [q[0], h, q[1]], D = [p[0], h, p[1]];
        var f = .72 + .22 * Math.abs(Math.cos(Math.atan2(q[1] - p[1], q[0] - p[0])));
        quad(A, B, C, f); quad(A, C, D, f);
      });
      var flat = ring.map(function (p) { return new THREE.Vector2(p[0], p[1]); });
      THREE.ShapeUtils.triangulateShape(flat, []).forEach(function (t) {
        var top = t.map(function (n) { return [ring[n][0], h, ring[n][1]]; });
        quad(top[0], top[1], top[2], 1.05);
      });
    });
    var city = new THREE.BufferGeometry();
    city.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    city.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    scene.add(new THREE.Mesh(city, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));

    // Overhead bridges: the start line and a few along the lap
    var banner = bannerTexture(THREE, name.toUpperCase());
    var bridge = function (k) {
      var p = track.pts[k], t = Math.atan2(track.normal[k][1], track.normal[k][0]);
      var g = new THREE.Group(), span = HALF_WIDTH * 2 + 8;
      var deck = new THREE.Mesh(new THREE.BoxGeometry(span, 2.4, 1), [lit({ color: 0x17623f }), lit({ color: 0x17623f }), lit({ color: 0x223 }), lit({ color: 0x223 }), lit({ map: banner }), lit({ map: banner })]);
      deck.position.y = 8.2;
      g.add(deck);
      [-1, 1].forEach(function (s) {
        var post = new THREE.Mesh(new THREE.BoxGeometry(.7, 8, .7), lit({ color: 0x2c3036 }));
        post.position.set(s * span / 2, 4, 0);
        g.add(post);
      });
      g.position.set(p[0], 0, p[1]);
      g.rotation.y = -t;
      scene.add(g);
    };
    bridge(10);
    [0.22, 0.46, 0.7, 0.88].forEach(function (f) { bridge(Math.floor(N * f)); });

    // Chequered line and the grandstands along the start
    var check = canvasTexture(THREE, 64, 8, function (g, w, h) {
      for (var x = 0; x < w; x += 4) for (var y = 0; y < h; y += 4) { g.fillStyle = ((x + y) / 4) % 2 ? '#fff' : '#111'; g.fillRect(x, y, 4, 4); }
    });
    var line = new THREE.Mesh(new THREE.PlaneGeometry(HALF_WIDTH * 2, 2), lit({ map: check }));
    line.rotation.x = -Math.PI / 2;
    var ls = track.pts[0], lt = Math.atan2(track.normal[0][1], track.normal[0][0]);
    line.rotation.z = -lt;
    line.position.set(ls[0], 0.04, ls[1]);
    scene.add(line);

    var crowd = canvasTexture(THREE, 1024, 256, function (g, w, h) {
      g.fillStyle = '#39404c'; g.fillRect(0, 0, w, h);
      var cols = ['#e8e4d8', '#d0473f', '#3e6db5', '#e0b83f', '#252a33', '#7f8794'];
      for (var i = 0; i < 22000; i++) { g.fillStyle = cols[Math.floor(Math.random() * cols.length)]; g.fillRect(Math.random() * w, Math.random() * h, 2, 2); }
    }, true);
    for (var s = -60; s <= 60; s += 15) {
      var at = (s + N) % N, q = track.pts[at], nn = track.normal[at], off2 = HALF_WIDTH + 44;
      [-1, 1].forEach(function (side) {
        var sx = q[0] + nn[0] * side * off2, sz = q[1] + nn[1] * side * off2;
        if (track.clearance(sx, sz, at) < off2) return;
        var tribune = new THREE.Mesh(new THREE.BoxGeometry(50, 10, 14), [lit({ color: 0x4a505a }), lit({ color: 0x4a505a }), lit({ color: 0xc9ccd2 }), lit({ color: 0x4a505a }), lit({ map: crowd }), lit({ map: crowd })]);
        tribune.position.set(sx, 5, sz);
        tribune.rotation.y = -Math.atan2(nn[1], nn[0]);
        scene.add(tribune);
      });
    }
  }

  // ---------- cockpit ----------

  function tube(THREE, points, radius, material, closed) {
    var curve = new THREE.CatmullRomCurve3(points.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); }));
    return new THREE.Mesh(new THREE.TubeGeometry(curve, 40, radius, 10, !!closed), material);
  }

  function buildCockpit(THREE, camera) {
    var group = new THREE.Group();
    camera.add(group);

    var carbonTex = canvasTexture(THREE, 32, 32, function (g, w, h) {
      g.fillStyle = '#17181b'; g.fillRect(0, 0, w, h);
      g.fillStyle = '#1e2023';
      for (var x = 0; x < w; x += 8) for (var y = 0; y < h; y += 8) if ((x + y) % 16 === 0) g.fillRect(x, y, 8, 8);
    }, true);
    carbonTex.repeat.set(40, 40);
    var carbon = new THREE.MeshPhongMaterial({ map: carbonTex, shininess: 30, specular: 0x222222 });
    // Team colours and race number only: no sponsor or team logos.
    var paint = new THREE.MeshPhongMaterial({ color: 0x121f45, shininess: 18, specular: 0x222233 });
    var noseTex = canvasTexture(THREE, 128, 512, function (g, w, h) {
      g.fillStyle = '#121f45'; g.fillRect(0, 0, w, h);
      g.fillStyle = '#d8102a'; g.fillRect(0, 0, w, 90);
      g.fillStyle = '#f6c600'; g.fillRect(0, 90, w, 26);
      g.fillStyle = '#ffffff'; g.font = '700 110px Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText('3', w / 2, 240);
    });
    var rubber = new THREE.MeshLambertMaterial({ color: 0x151517 });

    // Halo
    group.add(tube(THREE, [[-.46, -.2, .05], [-.42, .08, -.3], [-.2, .2, -.56], [0, .22, -.63], [.2, .2, -.56], [.42, .08, -.3], [.46, -.2, .05]], .032, carbon));
    group.add(tube(THREE, [[0, .22, -.63], [0, .06, -.78], [0, -.2, -.9]], .03, carbon));

    // Cockpit rim and bodywork
    var rim = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 18), carbon);
    rim.scale.set(.62, .24, .78);
    rim.position.set(0, -.55, -.62);
    group.add(rim);
    var nose = new THREE.Mesh(new THREE.CylinderGeometry(.09, .34, 2.5, 18), new THREE.MeshPhongMaterial({ map: noseTex, shininess: 18 }));
    nose.rotation.x = -Math.PI / 2 - .05;
    nose.position.set(0, -.52, -2.1);
    nose.scale.z = .6;
    group.add(nose);
    var wing = new THREE.Mesh(new THREE.BoxGeometry(1.9, .03, .4), new THREE.MeshPhongMaterial({ color: 0xd8102a, shininess: 18 }));
    wing.position.set(0, -.86, -3.25);
    group.add(wing);

    // Front wheels and arms
    var sidewall = canvasTexture(THREE, 128, 128, function (g, w, h) {
      g.fillStyle = '#121214'; g.fillRect(0, 0, w, h);
      g.strokeStyle = '#ecece8'; g.lineWidth = 5;
      g.beginPath(); g.arc(w / 2, h / 2, 46, 0, Math.PI * 2); g.stroke();
      g.fillStyle = '#2b2d31'; g.beginPath(); g.arc(w / 2, h / 2, 30, 0, Math.PI * 2); g.fill();
    });
    var side = new THREE.MeshLambertMaterial({ map: sidewall });
    var wheels = [];
    [-1, 1].forEach(function (s) {
      var pivot = new THREE.Group();
      pivot.position.set(s * .86, -.72, -1.95);
      var tyre = new THREE.Mesh(new THREE.CylinderGeometry(.36, .36, .36, 28), [rubber, side, side]);
      tyre.rotation.z = Math.PI / 2;
      pivot.add(tyre);
      group.add(pivot);
      wheels.push({ pivot: pivot, tyre: tyre });
      group.add(tube(THREE, [[s * .12, -.5, -1.8], [s * .5, -.6, -1.92], [s * .72, -.68, -1.95]], .012, carbon));
      group.add(tube(THREE, [[s * .12, -.62, -2.2], [s * .5, -.68, -2.08], [s * .72, -.74, -1.97]], .012, carbon));
    });

    // Wheel with a live display
    var screen = document.createElement('canvas');
    screen.width = 256; screen.height = 128;
    var screenTex = new THREE.CanvasTexture(screen);
    var wheel = new THREE.Group();
    wheel.position.set(0, -.25, -.46);
    wheel.rotation.x = .35;
    var plate = new THREE.Mesh(new THREE.BoxGeometry(.3, .15, .025), carbon);
    wheel.add(plate);
    var display = new THREE.Mesh(new THREE.PlaneGeometry(.13, .065), new THREE.MeshBasicMaterial({ map: screenTex, toneMapped: false }));
    display.position.set(0, .012, .014);
    wheel.add(display);
    [-1, 1].forEach(function (s) {
      var grip = new THREE.Mesh(new THREE.CylinderGeometry(.024, .024, .14, 12), rubber);
      grip.position.set(s * .165, 0, 0);
      wheel.add(grip);
    });
    group.add(wheel);

    return { wheel: wheel, wheels: wheels, screen: screen, screenTex: screenTex };
  }

  function gearFor(speed) {
    return speed < 1 ? 'N' : String(Math.min(8, 1 + Math.floor(speed / 11.5)));
  }

  // ---------- game ----------

  function loadThree(done, fail) {
    if (root.THREE) return done();
    var s = document.createElement('script');
    s.src = THREE_URL;
    s.onload = done;
    s.onerror = fail;
    document.head.appendChild(s);
  }

  function open(btn, real) {
    var id = btn.getAttribute('data-circuit');
    var name = btn.getAttribute('data-name');
    var label = btn.getAttribute('data-label') || name;
    var path = btn.getAttribute('data-path');
    var viewbox = btn.getAttribute('data-viewbox');
    var THREE = root.THREE;

    var sim = createSim(path, LENGTHS[id]);
    var storeKey = 'trackBest:' + id;
    try { var saved = parseFloat(localStorage.getItem(storeKey)); if (saved > 0) sim.bestLap = saved; } catch (e) {}
    sim.onLap = function (t, valid) {
      if (!valid || t > sim.bestLap) return;
      try { localStorage.setItem(storeKey, String(t)); } catch (e) {}
    };

    var box = document.createElement('div');
    box.className = 'track-game';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', 'Drive ' + name);
    var revs = '';
    for (var r = 0; r < 24; r++) revs += '<i></i>';
    box.innerHTML =
      '<canvas></canvas>' +
      '<div class="tg-revs">' + revs + '</div>' +
      '<div class="tg-times"><b class="tg-now"></b><span class="tg-last"></span><span class="tg-best"></span><span class="tg-lap"></span></div>' +
      '<div class="tg-note"></div>' +
      '<div class="tg-speed"><b></b><small>km/h</small><em></em></div>' +
      '<svg class="tg-map" viewBox="' + viewbox + '" aria-hidden="true"><path d="' + path + '"/><circle r="9"/></svg>' +
      '<div class="tg-driver"><b>Max Verstappen</b> · No. 3 · Red Bull colours</div>' +
      '<div class="tg-help">↑ throttle · ↓ brake · ← → steer · R restart · Esc close</div>' +
      '<div class="tg-credit">Map data © OpenStreetMap contributors</div>' +
      '<button type="button" class="tg-close">Close</button>';
    document.body.appendChild(box);
    document.documentElement.style.overflow = 'hidden';

    var q = function (s) { return box.querySelector(s); };
    var canvas = q('canvas');
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, logarithmicDepthBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    var scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xdde8ee, 300, 1900);
    scene.add(new THREE.HemisphereLight(0xdcebff, 0x55663a, .85));
    var sun = new THREE.DirectionalLight(0xfff4e0, .75);
    sun.position.set(.5, 1, .3);
    scene.add(sun);

    var camera = new THREE.PerspectiveCamera(72, 1, .04, 3400);
    camera.rotation.order = 'YXZ';
    scene.add(camera);

    var track = analyseTrack(sim.pts);
    buildWorld(THREE, scene, track, label, real);
    var cockpit = buildCockpit(THREE, camera);
    var mapDot = q('.tg-map circle');
    // The outline is stretched to the lap length, so scale the car's position back for the mini map.
    var parsed = parsePath(path), rawLen = 0;
    parsed.forEach(function (p, i) { var o = parsed[(i + 1) % parsed.length]; rawLen += Math.hypot(o[0] - p[0], o[1] - p[1]); });
    var mapScale = rawLen / (LENGTHS[id] || 5000);

    var resize = function () {
      var w = box.clientWidth, h = box.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener('resize', resize);

    var input = { up: false, down: false, left: false, right: false };
    var keys = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', w: 'up', s: 'down', a: 'left', d: 'right' };
    var closed = false;
    var close = function () {
      closed = true;
      window.removeEventListener('keydown', down, true);
      window.removeEventListener('keyup', up, true);
      window.removeEventListener('blur', release);
      window.removeEventListener('resize', resize);
      document.documentElement.style.overflow = '';
      renderer.dispose();
      box.remove();
      btn.focus();
      root.TrackGame.current = null;
    };
    var down = function (e) {
      var k = keys[e.key];
      if (k) { input[k] = true; e.preventDefault(); }
      else if (e.key === 'r' || e.key === 'R') sim.restart();
      else if (e.key === 'Escape') close();
    };
    var up = function (e) { var k = keys[e.key]; if (k) { input[k] = false; e.preventDefault(); } };
    var release = function () { input.up = input.down = input.left = input.right = false; };
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    window.addEventListener('blur', release);
    q('.tg-close').addEventListener('click', close);
    root.TrackGame.current = { sim: sim, input: input };

    var revCells = box.querySelectorAll('.tg-revs i');
    var now = q('.tg-now'), lastEl = q('.tg-last'), bestEl = q('.tg-best'), lapEl = q('.tg-lap'), note = q('.tg-note');
    var speedEl = q('.tg-speed b'), gearEl = q('.tg-speed em');
    var roll = 0, wheelSpin = 0, screenTimer = 0;
    var last = performance.now();

    (function frame(t) {
      if (closed) return;
      var dt = Math.min(0.05, (t - last) / 1000);
      last = t;
      sim.step(dt, input);

      var abs = Math.abs(sim.speed), frac = abs / TOP_SPEED;
      var buzz = sim.offTrack ? .012 * Math.min(1, abs / 15) : .0012 * frac;
      roll += (-sim.steer * .035 * frac - roll) * Math.min(1, dt * 5);
      camera.position.set(sim.x, .92 + (Math.random() - .5) * buzz, sim.y);
      camera.rotation.set((Math.random() - .5) * buzz * .5, -sim.heading - Math.PI / 2 - sim.steer * .05, roll);
      var fov = 70 + 8 * frac;
      if (Math.abs(camera.fov - fov) > .05) { camera.fov = fov; camera.updateProjectionMatrix(); }
      scene.userData.sky.position.set(sim.x, 0, sim.y);

      cockpit.wheel.rotation.z = -sim.steer * 1.3;
      wheelSpin += sim.speed / .36 * dt;
      cockpit.wheels.forEach(function (w) {
        w.pivot.rotation.y = -sim.steer * .3;
        w.tyre.rotation.x = wheelSpin;
      });

      // Dashboard text doesn't need to redraw every frame.
      screenTimer -= dt;
      if (screenTimer <= 0) {
        screenTimer = .08;
        var g = cockpit.screen.getContext('2d');
        g.fillStyle = '#05070a'; g.fillRect(0, 0, 256, 128);
        g.textAlign = 'center'; g.fillStyle = '#fff';
        g.font = '700 74px monospace'; g.fillText(gearFor(abs), 128, 84);
        g.font = '700 24px monospace'; g.fillStyle = '#6fe08a'; g.fillText(String(Math.round(abs * 3.6)), 46, 72);
        g.font = '700 18px monospace'; g.fillStyle = '#9fb4ff'; g.fillText('VER 3', 128, 112);
        g.font = '700 24px monospace';
        g.fillStyle = '#ffd23f'; g.fillText(sim.bestLap ? fmt(sim.bestLap).slice(0, 6) : '-', 208, 72);
        cockpit.screenTex.needsUpdate = true;

        now.textContent = fmt(sim.lapTime);
        lastEl.textContent = 'Last ' + fmt(sim.lastLap) + (sim.lastLap !== null && !sim.lapValid ? ' ✕' : '');
        bestEl.textContent = 'Best ' + fmt(sim.bestLap);
        lapEl.textContent = 'Lap ' + (sim.laps + 1);
        speedEl.textContent = Math.round(abs * 3.6);
        gearEl.textContent = gearFor(abs);
        q('.tg-help').style.opacity = sim.running ? 0 : 1;
        note.textContent = !sim.running ? 'Press ↑ to start' : sim.wrongWay ? 'Wrong way' : sim.offTrack ? 'Off track' : '';
        var within = (abs % 11.5) / 11.5, lit = abs < 1 ? 0 : Math.ceil((abs >= 80.5 ? frac : within) * revCells.length);
        revCells.forEach(function (c, i) { c.className = i < lit ? 'on' : ''; });
        mapDot.setAttribute('cx', sim.x * mapScale);
        mapDot.setAttribute('cy', sim.y * mapScale);
      }

      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    })(last);
  }

  root.TrackGame = { createSim: createSim, fmt: fmt, TOP_SPEED: TOP_SPEED, current: null };
  if (typeof document !== 'undefined') {
    document.querySelectorAll('[data-track-game]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var text = btn.textContent;
        btn.textContent = 'Loading…';
        btn.disabled = true;
        var ready = function () { btn.textContent = text; btn.disabled = false; };
        var failed = function () { btn.textContent = 'Could not load the track'; };
        loadThree(function () {
          fetch(btn.getAttribute('data-scene')).then(function (r) { return r.json(); }).then(function (real) {
            ready();
            try { open(btn, real); } catch (e) { btn.textContent = 'This browser cannot run the 3D view'; }
          }).catch(failed);
        }, failed);
      });
    });
  }
})(typeof window !== 'undefined' ? window : globalThis);
