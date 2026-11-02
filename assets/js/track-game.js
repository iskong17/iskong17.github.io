/* Drive a circuit from its page: a top-down arcade lap on the real track outline.
   The outline's scale isn't stored, so every track is stretched to the same length:
   lap times compare only with your own laps on that circuit, not with real ones. */
(function (root) {
  'use strict';

  var TRACK_LENGTH = 5000; // metres, for every circuit
  var HALF_WIDTH = 13;
  var SPACING = 5;
  var TOP_SPEED = 92;
  var OFF_SPEED = 32;

  function parsePath(d) {
    var nums = d.match(/-?\d+(\.\d+)?/g).map(Number);
    var pts = [];
    for (var i = 0; i + 1 < nums.length; i += 2) pts.push([nums[i], nums[i + 1]]);
    return pts;
  }

  // Close the lap and resample it evenly, with a light smoothing of the position noise.
  function buildTrack(d) {
    var raw = parsePath(d);
    var i, n = raw.length, len = 0;
    for (i = 0; i < n; i++) {
      var a = raw[i], b = raw[(i + 1) % n];
      len += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    var scale = TRACK_LENGTH / len;
    var count = Math.round(TRACK_LENGTH / SPACING);
    var pts = [], seg = 0, into = 0;
    var segLen = function (k) { var p = raw[k], q = raw[(k + 1) % n]; return Math.hypot(q[0] - p[0], q[1] - p[1]) * scale; };
    for (i = 0; i < count; i++) {
      while (into > segLen(seg) && seg < n - 1) { into -= segLen(seg); seg++; }
      var p = raw[seg], q = raw[(seg + 1) % n], t = segLen(seg) ? Math.min(1, into / segLen(seg)) : 0;
      pts.push([(p[0] + (q[0] - p[0]) * t) * scale, (p[1] + (q[1] - p[1]) * t) * scale]);
      into += TRACK_LENGTH / count;
    }
    for (var pass = 0; pass < 3; pass++) {
      pts = pts.map(function (_, k) {
        var s = [0, 0];
        for (var o = -3; o <= 3; o++) { var v = pts[(k + o + count) % count]; s[0] += v[0]; s[1] += v[1]; }
        return [s[0] / 7, s[1] / 7];
      });
    }
    return pts;
  }

  // The driving model, free of the page so it can be stepped on its own.
  function createSim(d) {
    var pts = buildTrack(d), N = pts.length;
    var sim = {
      pts: pts, x: 0, y: 0, heading: 0, speed: 0, idx: 0, progress: 0, offTrack: false, wrongWay: false,
      running: false, lapTime: 0, offTime: 0, lastLap: null, bestLap: null, lapValid: true, laps: 0, onLap: null
    };
    function tangent(i) {
      var a = pts[(i - 1 + N) % N], b = pts[(i + 1) % N];
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
      sim.speed = 0; sim.offTrack = false; sim.wrongWay = false;
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

      var grip = sim.offTrack ? 12 : 40;
      var abs = Math.abs(sim.speed);
      var rate = Math.min(1.9, grip / Math.max(abs, 1)) * Math.min(1, abs / 4);
      var steer = (input.right ? 1 : 0) - (input.left ? 1 : 0);
      sim.heading += steer * rate * (sim.speed < 0 ? -1 : 1) * dt;
      sim.x += Math.cos(sim.heading) * sim.speed * dt;
      sim.y += Math.sin(sim.heading) * sim.speed * dt;

      var at = nearest(sim.idx, 50);
      var delta = at.i - sim.idx;
      if (delta > N / 2) delta -= N;
      if (delta < -N / 2) delta += N;
      sim.idx = at.i;
      sim.progress += delta;
      sim.offTrack = at.dist > HALF_WIDTH;
      if (sim.offTrack && sim.running) sim.offTime += dt;
      if (at.dist > 90) sim.reset(at.i); // lost in the infield: back on the line, without the speed

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

  // Everything drawn for one frame. `view` is {w, h, dpr}.
  function draw(ctx, sim, view, cache) {
    var w = view.w, h = view.h;
    ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
    ctx.fillStyle = '#2f6b3a';
    ctx.fillRect(0, 0, w, h);

    var zoom = 3.4 - 1.5 * Math.min(1, Math.abs(sim.speed) / TOP_SPEED);
    ctx.save();
    ctx.translate(w / 2, h * 0.68);
    ctx.rotate(-sim.heading - Math.PI / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-sim.x, -sim.y);

    // Grass tiles give a sense of speed away from the kerbs.
    if (cache.grass) {
      ctx.fillStyle = cache.grass;
      ctx.fillRect(sim.x - 700, sim.y - 700, 1400, 1400);
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#f2f2f2'; ctx.lineWidth = HALF_WIDTH * 2 + 3; ctx.stroke(cache.path);
    ctx.setLineDash([6, 6]); ctx.lineCap = 'butt';
    ctx.strokeStyle = '#d6201f'; ctx.stroke(cache.path);
    ctx.setLineDash([]); ctx.lineCap = 'round';
    ctx.strokeStyle = '#3a3d44'; ctx.lineWidth = HALF_WIDTH * 2; ctx.stroke(cache.path);
    ctx.setLineDash([5, 11]); ctx.strokeStyle = 'rgba(255,255,255,.22)'; ctx.lineWidth = .5; ctx.stroke(cache.path);
    ctx.setLineDash([]);

    // Start line, across the track at the first point.
    var s = sim.pts[0], ang = sim.tangent(0);
    ctx.save();
    ctx.translate(s[0], s[1]);
    ctx.rotate(ang);
    for (var row = 0; row < 2; row++) {
      for (var k = -7; k < 7; k++) {
        ctx.fillStyle = (row + k + 14) % 2 ? '#fff' : '#111';
        ctx.fillRect(row * 2 - 2, k * HALF_WIDTH / 7, 2, HALF_WIDTH / 7);
      }
    }
    ctx.restore();

    // The car, nose along its heading.
    ctx.save();
    ctx.translate(sim.x, sim.y);
    ctx.rotate(sim.heading);
    ctx.scale(2.1, 2.1); // larger than life, so it reads at speed
    ctx.fillStyle = '#111';
    [[1.3, 1], [1.3, -1], [-1.5, 1.05], [-1.5, -1.05]].forEach(function (t) { ctx.fillRect(t[0] - .4, t[1] - .25, .8, .5); });
    ctx.fillStyle = '#16225a'; ctx.fillRect(-2.2, -.55, 4.4, 1.1);
    ctx.fillStyle = '#e10600'; ctx.fillRect(2.2, -1, .4, 2); ctx.fillRect(-2.6, -.95, .4, 1.9);
    ctx.fillStyle = '#f5c518'; ctx.fillRect(-.3, -.32, 1, .64);
    ctx.restore();
    ctx.restore();

    minimap(ctx, sim, view, cache);
    hud(ctx, sim, view);
  }

  function minimap(ctx, sim, view, cache) {
    var size = Math.min(170, view.w * 0.26), m = cache.bounds;
    var f = size / Math.max(m.w, m.h);
    var ox = view.w - size - 14 + (size - m.w * f) / 2, oy = 14 + (size - m.h * f) / 2;
    ctx.fillStyle = 'rgba(0,0,0,.45)';
    ctx.fillRect(view.w - size - 22, 6, size + 16, size + 16);
    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(f, f);
    ctx.translate(-m.x, -m.y);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.2 / f; ctx.stroke(cache.path);
    ctx.fillStyle = '#e10600';
    ctx.beginPath(); ctx.arc(sim.x, sim.y, 4.5 / f, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function hud(ctx, sim, view) {
    ctx.fillStyle = 'rgba(0,0,0,.55)';
    ctx.fillRect(10, 10, 196, 96);
    ctx.fillStyle = '#fff';
    ctx.font = '700 26px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(fmt(sim.lapTime), 20, 40);
    ctx.font = '13px "JetBrains Mono", monospace';
    ctx.fillText('Last  ' + fmt(sim.lastLap) + (sim.lastLap !== null && !sim.lapValid ? ' ✕' : ''), 20, 64);
    ctx.fillText('Best  ' + fmt(sim.bestLap), 20, 82);
    ctx.fillText('Lap   ' + (sim.laps + 1), 20, 98);
    ctx.textAlign = 'center';
    ctx.font = '700 34px "JetBrains Mono", monospace';
    ctx.fillText(String(Math.round(Math.abs(sim.speed) * 3.6)), view.w / 2, view.h - 34);
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText('km/h', view.w / 2, view.h - 16);
    var note = !sim.running ? 'Press ↑ to start' : sim.wrongWay ? 'WRONG WAY' : sim.offTrack ? 'OFF TRACK' : '';
    if (note) {
      ctx.font = '700 22px "JetBrains Mono", monospace';
      ctx.fillStyle = sim.running ? '#ffd23f' : '#fff';
      ctx.fillText(note, view.w / 2, 58);
    }
    ctx.textAlign = 'left';
  }

  function makeCache(ctx, sim) {
    var path = new Path2D(), minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    sim.pts.forEach(function (p, i) {
      if (i) path.lineTo(p[0], p[1]); else path.moveTo(p[0], p[1]);
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    });
    path.closePath();
    var tile = ctx.canvas.ownerDocument ? ctx.canvas.ownerDocument.createElement('canvas') : null;
    var grass = null;
    if (tile) {
      tile.width = tile.height = 80;
      var t = tile.getContext('2d');
      t.fillStyle = '#2f6b3a'; t.fillRect(0, 0, 80, 80);
      t.fillStyle = '#2a6235'; t.fillRect(0, 0, 40, 40); t.fillRect(40, 40, 40, 40);
      grass = ctx.createPattern(tile, 'repeat');
    }
    return { path: path, grass: grass, bounds: { x: minX, y: minY, w: maxX - minX, h: maxY - minY } };
  }

  function open(btn) {
    var id = btn.getAttribute('data-circuit');
    var name = btn.getAttribute('data-name');
    var sim = createSim(btn.getAttribute('data-path'));
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
    box.innerHTML = '<div class="track-game-bar"><b></b><span>↑ throttle · ↓ brake · ← → steer · R restart · Esc close</span><button type="button">Close</button></div><canvas></canvas>';
    box.querySelector('b').textContent = name;
    document.body.appendChild(box);
    document.documentElement.style.overflow = 'hidden';

    var canvas = box.querySelector('canvas'), ctx = canvas.getContext('2d');
    var cache = makeCache(ctx, sim), view = { w: 0, h: 0, dpr: 1 };
    var resize = function () {
      view.dpr = window.devicePixelRatio || 1;
      view.w = canvas.clientWidth; view.h = canvas.clientHeight;
      canvas.width = view.w * view.dpr; canvas.height = view.h * view.dpr;
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
      box.remove();
      btn.focus();
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
    box.querySelector('button').addEventListener('click', close);

    var last = performance.now();
    (function frame(now) {
      if (closed) return;
      var dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      sim.step(dt, input);
      draw(ctx, sim, view, cache);
      requestAnimationFrame(frame);
    })(last);
  }

  root.TrackGame = { createSim: createSim, draw: draw, makeCache: makeCache, fmt: fmt, TOP_SPEED: TOP_SPEED };
  if (typeof document !== 'undefined') {
    document.querySelectorAll('[data-track-game]').forEach(function (btn) {
      btn.addEventListener('click', function () { open(btn); });
    });
  }
})(typeof window !== 'undefined' ? window : globalThis);
