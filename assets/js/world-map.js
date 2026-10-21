/* Interactive world map shared by the Gallery and Race > Circuits sections.
   Needs d3 and topojson-client. Countries in `highlight` are filled; `markers`
   are pins with lat/lng. Clicking a country zooms to it and calls onSelect. */
(function () {
  var ATLAS = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json';
  var atlasPromise = null;

  // Names used in posts and in the F1 data that differ from Natural Earth names.
  var ALIASES = {
    'usa': 'united states of america', 'united states': 'united states of america', 'us': 'united states of america',
    'uk': 'united kingdom', 'great britain': 'united kingdom', 'england': 'united kingdom',
    'uae': 'united arab emirates', 'korea': 'south korea', 'republic of korea': 'south korea',
    'czech republic': 'czechia'
  };

  function norm(name) {
    var n = String(name || '').trim().toLowerCase();
    return ALIASES[n] || n;
  }

  // Natural Earth draws France with its overseas departments (e.g. French Guiana).
  // Keep the European part so highlighting France does not paint South America.
  function mainland(f) {
    if (f.properties.name !== 'France' || f.geometry.type !== 'MultiPolygon') return f;
    var polys = f.geometry.coordinates.filter(function (poly) {
      var lon = poly[0][0][0], lat = poly[0][0][1];
      return lon > -10 && lon < 20 && lat > 35;
    });
    return { type: 'Feature', id: f.id, properties: f.properties, geometry: { type: 'MultiPolygon', coordinates: polys } };
  }

  function loadAtlas() {
    if (!atlasPromise) {
      atlasPromise = fetch(ATLAS).then(function (r) {
        if (!r.ok) throw new Error('atlas ' + r.status);
        return r.json();
      });
    }
    return atlasPromise;
  }

  var WorldMap = window.WorldMap = function (el, opts) {
    var width = 960, height = 480;
    var highlight = {};
    (opts.highlight || []).forEach(function (n) { highlight[norm(n)] = true; });
    var markers = (opts.markers || []).filter(function (m) {
      return m.lat != null && m.lng != null && m.lat !== '' && isFinite(m.lat) && isFinite(m.lng);
    });

    var svg = d3.select(el).append('svg')
      .attr('viewBox', '0 0 ' + width + ' ' + height)
      .attr('class', 'world-map-svg')
      .attr('role', 'img')
      .attr('aria-label', opts.label || 'World map');
    var g = svg.append('g');
    var projection = d3.geoNaturalEarth1().fitExtent([[6, 6], [width - 6, height - 6]], { type: 'Sphere' });
    var path = d3.geoPath(projection);
    var current = 1;

    var zoom = d3.zoom().scaleExtent([1, 60]).on('zoom', function (e) {
      current = e.transform.k;
      g.attr('transform', e.transform);
      g.selectAll('.map-country').attr('stroke-width', 0.6 / current);
      g.selectAll('.map-marker').attr('r', 5 / Math.sqrt(current)).attr('stroke-width', 1.5 / current);
    });
    svg.call(zoom).on('dblclick.zoom', null);

    var selected = null;

    function zoomToBounds(b) {
      var dx = b[1][0] - b[0][0], dy = b[1][1] - b[0][1];
      var cx = (b[0][0] + b[1][0]) / 2, cy = (b[0][1] + b[1][1]) / 2;
      var k = Math.max(1, Math.min(60, 0.8 / Math.max(dx / width, dy / height)));
      svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(k).translate(-cx, -cy));
    }

    function markersIn(name) {
      return markers.filter(function (m) { return norm(m.country) === name; });
    }

    function select(feature, name) {
      var inside = markersIn(name);
      g.selectAll('.map-country').classed('is-selected', function (d) { return d === feature; });
      selected = name;
      if (inside.length) {
        // Zoom to the pins, so overseas territories do not shrink the view.
        var pts = inside.map(function (m) { return projection([m.lng, m.lat]); });
        var xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
        var minSpan = 14;
        var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
        var fb = feature ? path.bounds(feature) : null;
        if (fb && (fb[1][0] - fb[0][0]) < 120 && (fb[1][1] - fb[0][1]) < 80) {
          x0 = Math.min(x0, fb[0][0]); x1 = Math.max(x1, fb[1][0]); y0 = Math.min(y0, fb[0][1]); y1 = Math.max(y1, fb[1][1]);
        }
        var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        var hw = Math.max((x1 - x0) / 2, minSpan), hh = Math.max((y1 - y0) / 2, minSpan / 2);
        zoomToBounds([[cx - hw, cy - hh], [cx + hw, cy + hh]]);
      } else if (feature) {
        zoomToBounds(path.bounds(feature));
      }
      if (opts.onSelect) opts.onSelect(feature ? feature.properties.name : name, inside);
    }

    function reset() {
      selected = null;
      g.selectAll('.map-country').classed('is-selected', false);
      svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity);
      if (opts.onSelect) opts.onSelect(null, markers);
    }

    var resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'map-reset';
    resetBtn.textContent = 'Reset';
    resetBtn.addEventListener('click', reset);
    el.appendChild(resetBtn);

    var countries = [];
    var ready = loadAtlas().then(function (topo) {
      countries = topojson.feature(topo, topo.objects.countries).features.map(mainland);
      g.append('path').datum({ type: 'Sphere' }).attr('class', 'map-sphere').attr('d', path).on('click', reset);
      g.selectAll('.map-country')
        .data(countries)
        .join('path')
        .attr('class', function (d) { return 'map-country' + (highlight[norm(d.properties.name)] ? ' is-on' : ''); })
        .attr('d', path)
        .on('click', function (e, d) {
          e.stopPropagation();
          var name = norm(d.properties.name);
          if (selected === name) reset(); else select(d, name);
        })
        .append('title').text(function (d) { return d.properties.name; });

      g.selectAll('.map-marker')
        .data(markers)
        .join('circle')
        .attr('class', function (m) { return 'map-marker' + (m.dim ? ' is-dim' : ''); })
        .attr('cx', function (m) { return projection([m.lng, m.lat])[0]; })
        .attr('cy', function (m) { return projection([m.lng, m.lat])[1]; })
        .attr('r', 5)
        .on('click', function (e, m) {
          e.stopPropagation();
          var name = norm(m.country);
          var feature = countries.filter(function (d) { return norm(d.properties.name) === name; })[0] || null;
          select(feature, name);
        })
        .append('title').text(function (m) { return m.label; });

      if (opts.onSelect) opts.onSelect(null, markers);
    }).catch(function () {
      el.classList.add('is-failed');
    });

    function featureOf(name) {
      return countries.filter(function (d) { return norm(d.properties.name) === name; })[0] || null;
    }

    return {
      reset: reset,
      // Select a country by name, as if it was clicked on the map.
      selectCountry: function (name) {
        ready.then(function () { var n = norm(name); select(featureOf(n), n); });
      },
      // Zoom to one marker and report only that marker.
      focusMarker: function (m) {
        ready.then(function () {
          var n = norm(m.country);
          var p = projection([m.lng, m.lat]);
          g.selectAll('.map-country').classed('is-selected', function (d) { return norm(d.properties.name) === n; });
          selected = n;
          zoomToBounds([[p[0] - 14, p[1] - 7], [p[0] + 14, p[1] + 7]]);
          if (opts.onSelect) opts.onSelect(m.country, [m]);
        });
      }
    };
  };
  WorldMap.norm = norm;
})();
