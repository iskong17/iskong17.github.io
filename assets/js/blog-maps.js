/* Wires the world maps on the blog page: Gallery (visited countries and
   geotagged posts) and Formula 1 > Circuits (circuits of the last five seasons). */
(function () {
  if (!window.WorldMap || !window.d3) return;

  function readJSON(id) {
    var el = document.getElementById(id);
    try { return el ? JSON.parse(el.textContent) : null; } catch (e) { return null; }
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function key(s) { return WorldMap.norm(s); }
  function backButton() { return '<button type="button" class="map-back" data-map-back>← All</button>'; }

  // ---------- Gallery ----------
  var gallery = readJSON('gallery-data');
  var galleryEl = document.getElementById('gallery-map');
  var galleryPanel = document.getElementById('gallery-panel');
  if (gallery && galleryEl && galleryPanel) {
    var places = gallery.places || [];
    var visited = (gallery.visited || []).map(function (v) { return typeof v === 'string' ? { country: v } : v; });
    // One row per country: notes from visited.yml plus the posts made there.
    var byCountry = {};
    visited.forEach(function (v) {
      var k = key(v.country);
      byCountry[k] = byCountry[k] || { country: v.country, notes: [], posts: [] };
      if (v.note) byCountry[k].notes.push(v.note);
    });
    places.forEach(function (p) {
      var k = key(p.country);
      byCountry[k] = byCountry[k] || { country: p.country, notes: [], posts: [] };
      byCountry[k].posts.push(p);
    });
    var rows = Object.keys(byCountry).map(function (k) { return byCountry[k]; })
      .sort(function (a, b) { return a.country.localeCompare(b.country); });
    var pins = places.concat(visited.filter(function (v) { return v.lat != null && v.lng != null; })
      .map(function (v) { return { label: v.note || v.country, country: v.country, lat: v.lat, lng: v.lng }; }));

    var galleryMap = WorldMap(galleryEl, {
      label: 'Countries I have visited',
      highlight: rows.map(function (r) { return r.country; }),
      markers: pins,
      onSelect: function (country) {
        if (!country) {
          galleryPanel.innerHTML =
            '<p class="map-kicker">Visited</p><p class="map-big">' + rows.length + (rows.length === 1 ? ' country' : ' countries') + '</p>' +
            '<div class="map-bars">' + rows.map(function (r) {
              var sub = r.notes.concat(r.posts.length ? [r.posts.length + (r.posts.length === 1 ? ' post' : ' posts')] : []).join(' · ');
              return '<button type="button" class="map-bar" data-country="' + esc(r.country) + '"><b>' + esc(r.country) + '</b>' +
                (sub ? '<span>' + esc(sub) + '</span>' : '') + '</button>';
            }).join('') + '</div>';
          return;
        }
        var row = byCountry[key(country)];
        var html = backButton() + '<p class="map-kicker">' + esc(country) + '</p>';
        if (row && row.notes.length) html += '<p class="map-hint">' + esc(row.notes.join(' · ')) + '</p>';
        if (row && row.posts.length) {
          html += '<ul class="map-list">' + row.posts.map(function (p) {
            return '<li><a href="' + esc(p.url) + '"><b>' + esc(p.label) + '</b><span>' + esc(p.place || '') + ' · ' + esc(p.date) + '</span></a></li>';
          }).join('') + '</ul>';
        } else if (!row) {
          html += '<p class="map-hint">Not visited yet.</p>';
        } else {
          html += '<p class="map-hint">No posts from here yet.</p>';
        }
        galleryPanel.innerHTML = html;
      }
    });
    galleryPanel.addEventListener('click', function (e) {
      var bar = e.target.closest('[data-country]');
      if (bar) galleryMap.selectCountry(bar.getAttribute('data-country'));
      if (e.target.closest('[data-map-back]')) galleryMap.reset();
    });
  }

  // ---------- Formula 1 circuits ----------
  var CONTINENT = {
    'uk': 'Europe', 'italy': 'Europe', 'monaco': 'Europe', 'spain': 'Europe', 'belgium': 'Europe', 'hungary': 'Europe',
    'austria': 'Europe', 'netherlands': 'Europe', 'france': 'Europe', 'portugal': 'Europe', 'germany': 'Europe',
    'russia': 'Europe', 'turkey': 'Europe', 'azerbaijan': 'Europe',
    'japan': 'Asia', 'china': 'Asia', 'singapore': 'Asia', 'malaysia': 'Asia', 'korea': 'Asia', 'india': 'Asia', 'vietnam': 'Asia',
    'bahrain': 'Middle East', 'saudi arabia': 'Middle East', 'uae': 'Middle East', 'qatar': 'Middle East',
    'usa': 'Americas', 'united states': 'Americas', 'canada': 'Americas', 'mexico': 'Americas', 'brazil': 'Americas', 'argentina': 'Americas',
    'australia': 'Oceania', 'south africa': 'Africa'
  };
  var ORDER = ['Europe', 'Middle East', 'Asia', 'Americas', 'Oceania', 'Africa', 'Other'];
  var SHOWN = 3;

  var circuitData = readJSON('circuit-data');
  var circuitEl = document.getElementById('circuit-map');
  var circuitPanel = document.getElementById('circuit-panel');
  if (circuitData && circuitEl && circuitPanel) {
    var pages = {};
    (circuitData.pages || []).forEach(function (p) { pages[p.id] = p; });
    var venues = (circuitData.venues || []).map(function (v) {
      var page = pages[v.id];
      var seasons = v.seasons || [];
      return {
        id: v.id, label: v.name, country: v.country, locality: v.locality, lat: v.lat, lng: v.lng,
        seasons: seasons, page: page || null, dim: !page,
        continent: CONTINENT[String(v.country).toLowerCase()] || 'Other'
      };
    });
    // Calendar circuits missing from the venue list still get a pin.
    Object.keys(pages).forEach(function (id) {
      if (!venues.some(function (v) { return v.id === id; })) {
        var p = pages[id];
        venues.push({ id: id, label: p.label, country: p.country, locality: p.locality, lat: p.lat, lng: p.lng, seasons: [], page: p, dim: false, continent: CONTINENT[String(p.country).toLowerCase()] || 'Other' });
      }
    });

    function card(v) {
      var p = v.page;
      var span = v.seasons.length ? (v.seasons.length > 1 ? v.seasons[0] + '–' + v.seasons[v.seasons.length - 1] : v.seasons[0]) : '';
      if (!p) {
        return '<div class="map-circuit is-past"><b>' + esc(v.label) + '</b><span>' + esc(v.locality) + '</span>' +
          (span ? '<span>Raced ' + esc(span) + ' · not on this year\'s calendar</span>' : '') + '</div>';
      }
      var track = p.path ? '<svg class="map-track" viewBox="' + esc(p.viewBox) + '" aria-hidden="true"><path d="' + esc(p.path) + '"/></svg>' : '';
      return '<a class="map-circuit" href="' + esc(p.url) + '">' + track +
        '<b>' + esc(p.label) + '</b>' +
        '<span>' + esc(p.locality) + ' · ' + p.races + ' GPs</span>' +
        (p.lastWinner ? '<span>Last winner: ' + esc(p.lastWinner) + ' (' + esc(p.lastSeason) + ')</span>' : '') +
        (p.record ? '<span>Fastest race lap: ' + esc(p.record) + ' · ' + esc(p.recordBy) + '</span>' : '') +
        '<span class="map-more">Circuit page →</span></a>';
    }

    function overview() {
      var groups = {};
      venues.forEach(function (v) { (groups[v.continent] = groups[v.continent] || []).push(v); });
      var current = venues.filter(function (v) { return v.page; }).length;
      var html = '<p class="map-kicker">Last five seasons</p><p class="map-big">' + venues.length + ' circuits</p>' +
        '<p class="map-hint">' + current + ' on this year\'s calendar. Grey pins are past venues.</p>';
      ORDER.forEach(function (c) {
        var list = (groups[c] || []).sort(function (a, b) { return (b.page ? 1 : 0) - (a.page ? 1 : 0) || a.label.localeCompare(b.label); });
        if (!list.length) return;
        html += '<div class="map-group"><p class="map-group-head">' + c + ' <span>' + list.length + '</span></p>' +
          list.map(function (v, i) {
            return '<button type="button" class="map-bar' + (v.page ? '' : ' is-past') + (i >= SHOWN ? ' is-extra' : '') + '" data-venue="' + esc(v.id) + '"' + (i >= SHOWN ? ' hidden' : '') + '>' +
              '<b>' + esc(v.label) + '</b><span>' + esc(v.country) + (v.page ? '' : ' · past') + '</span></button>';
          }).join('') +
          (list.length > SHOWN ? '<button type="button" class="map-group-more" data-more>More (' + (list.length - SHOWN) + ')</button>' : '') +
          '</div>';
      });
      return html;
    }

    var circuitMap = WorldMap(circuitEl, {
      label: 'Formula 1 circuits of the last five seasons',
      highlight: venues.map(function (v) { return v.country; }),
      markers: venues,
      onSelect: function (country, items) {
        if (!country) { circuitPanel.innerHTML = overview(); return; }
        circuitPanel.innerHTML = backButton() + '<p class="map-kicker">' + esc(country) + '</p>' +
          (items.length ? items.map(card).join('') : '<p class="map-hint">No Formula 1 race here in the last five seasons.</p>');
      }
    });

    circuitPanel.addEventListener('click', function (e) {
      var more = e.target.closest('[data-more]');
      if (more) {
        var group = more.closest('.map-group');
        var open = group.classList.toggle('is-open');
        group.querySelectorAll('.is-extra').forEach(function (b) { b.hidden = !open; });
        more.textContent = open ? 'Less' : 'More (' + group.querySelectorAll('.is-extra').length + ')';
        return;
      }
      var bar = e.target.closest('[data-venue]');
      if (bar) {
        var v = venues.filter(function (x) { return x.id === bar.getAttribute('data-venue'); })[0];
        if (v) circuitMap.focusMarker(v);
        return;
      }
      if (e.target.closest('[data-map-back]')) circuitMap.reset();
    });
  }
})();
