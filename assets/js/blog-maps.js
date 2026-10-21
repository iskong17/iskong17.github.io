/* Wires the world maps on the blog page: Gallery (visited countries and
   geotagged posts) and Race > Circuits (circuit locations and info). */
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

  // ---------- Gallery ----------
  var gallery = readJSON('gallery-data');
  var galleryEl = document.getElementById('gallery-map');
  var galleryPanel = document.getElementById('gallery-panel');
  if (gallery && galleryEl && galleryPanel) {
    var places = gallery.places || [];
    var visited = (gallery.visited || []).concat(places.map(function (p) { return p.country; }));
    var countries = {};
    visited.forEach(function (c) { countries[String(c).toLowerCase()] = true; });

    WorldMap(galleryEl, {
      label: 'Countries I have visited',
      highlight: visited,
      markers: places,
      onSelect: function (country, items) {
        if (!country) {
          galleryPanel.innerHTML =
            '<p class="map-kicker">Visited</p><p class="map-big">' + Object.keys(countries).length + (Object.keys(countries).length === 1 ? ' country' : ' countries') + '</p>' +
            '<p class="map-hint">나라를 누르면 확대되고, 그곳에서 올린 글이 여기에 표시됩니다.</p>';
          return;
        }
        var html = '<p class="map-kicker">' + esc(country) + '</p>';
        if (items.length) {
          html += '<ul class="map-list">' + items.map(function (p) {
            return '<li><a href="' + esc(p.url) + '"><b>' + esc(p.label) + '</b><span>' + esc(p.place || '') + ' · ' + esc(p.date) + '</span></a></li>';
          }).join('') + '</ul>';
        } else {
          html += '<p class="map-hint">아직 이곳의 글이 없습니다.</p>';
        }
        galleryPanel.innerHTML = html;
      }
    });
  }

  // ---------- Circuits ----------
  var circuits = readJSON('circuit-data');
  var circuitEl = document.getElementById('circuit-map');
  var circuitPanel = document.getElementById('circuit-panel');
  if (circuits && circuitEl && circuitPanel) {
    WorldMap(circuitEl, {
      label: 'Formula 1 circuits on the current calendar',
      highlight: circuits.map(function (c) { return c.country; }),
      markers: circuits,
      onSelect: function (country, items) {
        if (!country) {
          circuitPanel.innerHTML =
            '<p class="map-kicker">Calendar</p><p class="map-big">' + circuits.length + ' circuits</p>' +
            '<p class="map-hint">Click a country or a pin to see its circuits.</p>';
          return;
        }
        circuitPanel.innerHTML = '<p class="map-kicker">' + esc(country) + '</p>' + items.map(function (c) {
          var track = c.path ? '<svg class="map-track" viewBox="' + esc(c.viewBox) + '" aria-hidden="true"><path d="' + esc(c.path) + '"/></svg>' : '';
          return '<a class="map-circuit" href="' + esc(c.url) + '">' + track +
            '<b>' + esc(c.label) + '</b>' +
            '<span>' + esc(c.locality) + ' · ' + c.races + ' GPs</span>' +
            (c.lastWinner ? '<span>Last winner: ' + esc(c.lastWinner) + ' (' + esc(c.lastSeason) + ')</span>' : '') +
            (c.record ? '<span>Fastest race lap: ' + esc(c.record) + ' · ' + esc(c.recordBy) + '</span>' : '') +
            '</a>';
        }).join('') || '<p class="map-hint">No circuit here on the current calendar.</p>';
      }
    });
  }
})();
