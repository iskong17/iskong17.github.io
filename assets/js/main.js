(function () {
  var root = document.documentElement;

  // Theme toggle (dark is the default; the choice is remembered when storage works).
  var toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  }

  // Generic tab filter: buttons with data-filter inside `tabs`, items inside `list`.
  function setupFilter(tabs, items, matches, initial) {
    var buttons = tabs.querySelectorAll('[data-filter]');
    function apply(filter) {
      var last = null;
      items.forEach(function (item) {
        var show = matches(item, filter);
        item.classList.toggle('is-hidden', !show);
        item.classList.remove('is-last');
        if (show) last = item;
      });
      if (last) last.classList.add('is-last');
      buttons.forEach(function (b) {
        var on = b.getAttribute('data-filter') === filter;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
      });
    }
    buttons.forEach(function (b) {
      b.addEventListener('click', function () { apply(b.getAttribute('data-filter')); });
    });
    var known = Array.prototype.some.call(buttons, function (b) { return b.getAttribute('data-filter') === initial; });
    apply(known ? initial : buttons[0].getAttribute('data-filter'));
  }

  // Publications: Selected = first-author, Preprint = under review, Full = all.
  var pubTabs = document.querySelector('#publications .session-tabs');
  if (pubTabs) {
    setupFilter(pubTabs, document.querySelectorAll('#pub-list .pub'), function (row, f) {
      if (f === 'full') return true;
      if (f === 'preprint') return row.dataset.status === 'preprint';
      return row.dataset.selected === 'true';
    }, 'selected');
  }

  // Blog: category panels, Race sub-sections, and Study topics.
  // The URL hash keeps the view, e.g. #race, #race/circuits, #study/paper-review.
  var blogTabs = document.getElementById('blog-tabs');
  if (blogTabs) {
    var panels = document.querySelectorAll('.blog-panel');
    var raceTabs = document.getElementById('race-tabs');

    function activate(buttons, attr, value) {
      buttons.forEach(function (b) {
        var on = b.getAttribute(attr) === value;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
      });
    }
    function showPanel(name) {
      if (!document.querySelector('.blog-panel[data-panel="' + name + '"]')) name = 'all';
      panels.forEach(function (p) { p.hidden = p.getAttribute('data-panel') !== name; });
      activate(blogTabs.querySelectorAll('[data-panel]'), 'data-panel', name);
      return name;
    }
    function showRace(sub) {
      if (!raceTabs) return;
      if (!document.querySelector('.race-sub[data-sub="' + sub + '"]')) sub = 'seasons';
      document.querySelectorAll('.race-sub').forEach(function (s) { s.hidden = s.getAttribute('data-sub') !== sub; });
      activate(raceTabs.querySelectorAll('[data-sub]'), 'data-sub', sub);
    }
    // Study and Gallery filter by topic tab; Paper Review filters by subfield chip.
    function showTopic(panel, key) {
      var bySub = !!panel.querySelector('.subfield-chip');
      var attr = bySub ? 'data-subfield' : 'data-topic';
      var tabs = panel.querySelectorAll(bySub ? '.subfield-chip' : '.topic-tabs [data-topic]');
      if (!tabs.length) return;
      if (!panel.querySelector('[' + attr + '="' + key + '"]')) key = 'all';
      // Study's All view is a bar per topic; the flat grid shows only for one topic.
      var bars = panel.querySelector('.topic-bars');
      var grid = panel.querySelector('.sq-grid');
      if (bars) {
        bars.hidden = key !== 'all';
        if (grid) grid.hidden = key === 'all';
      }
      var cards = panel.querySelectorAll('.sq-grid > .sq-card, .paper-row');
      var any = !!bars && key === 'all';
      cards.forEach(function (c) {
        var show = key === 'all' || (bySub
          ? (' ' + (c.getAttribute('data-subfields') || '') + ' ').indexOf(' ' + key + ' ') !== -1
          : c.getAttribute('data-topic') === key);
        c.hidden = !show;
        if (show) any = true;
      });
      activate(tabs, attr, key);
      var empty = panel.querySelector('.topic-empty');
      if (!empty) {
        empty = document.createElement('p');
        empty.className = 'empty topic-empty';
        empty.textContent = 'No posts yet.';
        panel.appendChild(empty);
      }
      empty.hidden = any || cards.length === 0;
    }
    function route(hash) {
      var parts = (hash || '').replace(/^#/, '').split('/');
      var panel = showPanel(parts[0] || 'all');
      if (panel === 'race') showRace(parts[1] || 'seasons');
      var p = document.querySelector('.blog-panel[data-panel="' + panel + '"]');
      if (p) showTopic(p, parts[1] || 'all');
    }
    function go(hash) {
      history.replaceState(null, '', hash);
      route(hash);
    }

    blogTabs.querySelectorAll('[data-panel]').forEach(function (b) {
      b.addEventListener('click', function () { go('#' + b.getAttribute('data-panel')); });
    });
    document.querySelectorAll('[data-goto]').forEach(function (b) {
      b.addEventListener('click', function () { go('#' + b.getAttribute('data-goto')); window.scrollTo(0, blogTabs.offsetTop - 80); });
    });
    if (raceTabs) {
      raceTabs.querySelectorAll('[data-sub]').forEach(function (b) {
        b.addEventListener('click', function () { go('#race/' + b.getAttribute('data-sub')); });
      });
    }
    document.querySelectorAll('.topic-tabs [data-topic]').forEach(function (b) {
      var panel = b.closest('.blog-panel').getAttribute('data-panel');
      b.addEventListener('click', function () { go('#' + panel + '/' + b.getAttribute('data-topic')); });
    });
    document.querySelectorAll('.subfield-chip').forEach(function (b) {
      var panel = b.closest('.blog-panel').getAttribute('data-panel');
      b.addEventListener('click', function () {
        var sf = b.getAttribute('data-subfield');
        go('#' + panel + (sf === 'all' ? '' : '/' + sf));
      });
    });
    window.addEventListener('hashchange', function () { route(location.hash); });
    route(location.hash);
  }

  // Lightbox for publication figures and certificates.
  var lightbox = document.getElementById('lightbox');
  var lightboxImg = document.getElementById('lightbox-img');
  if (lightbox && lightboxImg) {
    function open(src, alt) {
      lightboxImg.src = src;
      lightboxImg.alt = alt || '';
      lightbox.classList.add('is-open');
    }
    document.querySelectorAll('img[data-zoom]').forEach(function (img) {
      img.addEventListener('click', function () { open(img.src, img.alt); });
    });
    document.querySelectorAll('[data-lightbox]').forEach(function (btn) {
      btn.addEventListener('click', function () { open(btn.getAttribute('data-lightbox'), btn.getAttribute('aria-label')); });
    });
    function close() { lightbox.classList.remove('is-open'); }
    lightbox.addEventListener('click', close);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
  }

  // Live F1 data for Max Verstappen from the Jolpica F1 API (Ergast-compatible).
  // The widget stays hidden if any request fails.
  var f1 = document.getElementById('f1-live');
  if (f1 && window.fetch) {
    var API = 'https://api.jolpi.ca/ergast/f1/current/';
    var DRIVER = 'max_verstappen';
    var CACHE_KEY = 'f1-live-v1';
    var CACHE_MS = 30 * 60 * 1000;

    function getJSON(path) {
      return fetch(API + path + '?format=json').then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      });
    }
    function set(name, text) {
      var el = f1.querySelector('[data-f1="' + name + '"]');
      if (el) el.textContent = text;
    }
    function render(d) {
      set('pos', 'P' + d.standing.position);
      set('pts', d.standing.points + ' pts · ' + d.standing.wins + (d.standing.wins === '1' ? ' win' : ' wins'));

      if (d.last) {
        set('last-name', d.last.raceName.replace('Grand Prix', 'GP'));
        var finished = /^\d+$/.test(d.last.positionText);
        set('last-pos', finished ? 'P' + d.last.positionText : 'DNF');
        set('last-detail', (d.last.grid === '0' ? 'Pit lane start' : 'Started P' + d.last.grid) + (finished ? '' : ' · ' + d.last.status));
      } else {
        set('last-pos', '–');
      }

      if (d.next) {
        set('next-name', d.next.raceName.replace('Grand Prix', 'GP'));
        var when = new Date(d.next.date + 'T' + (d.next.time || '00:00:00Z'));
        var days = Math.ceil((when - Date.now()) / 86400000);
        var date = when.toLocaleDateString('en-GB', { timeZone: 'Asia/Seoul', day: 'numeric', month: 'short' });
        set('next-when', date + (days > 0 ? ' · D-' + days : ' · Race week'));
      } else {
        set('next-name', 'Season over');
      }

      set('round', d.season + ' season · after round ' + d.round);
      f1.hidden = false;
    }

    var cached = null;
    try { cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null'); } catch (e) {}
    if (cached && Date.now() - cached.t < CACHE_MS) {
      render(cached.d);
    } else {
      Promise.all([
        getJSON('drivers/' + DRIVER + '/driverStandings/'),
        getJSON('last/results/'),
        getJSON('next/')
      ]).then(function (res) {
        var list = res[0].MRData.StandingsTable.StandingsLists[0];
        var race = res[1].MRData.RaceTable.Races[0];
        var next = res[2].MRData.RaceTable.Races[0];
        var mine = race && race.Results.filter(function (r) { return r.Driver.driverId === DRIVER; })[0];
        var d = {
          season: list.season,
          round: list.round,
          standing: list.DriverStandings[0],
          last: mine ? { raceName: race.raceName, positionText: mine.positionText, grid: mine.grid, status: mine.status } : null,
          next: next ? { raceName: next.raceName, date: next.date, time: next.time } : null
        };
        render(d);
        try { localStorage.setItem(CACHE_KEY, JSON.stringify({ t: Date.now(), d: d })); } catch (e) {}
      }).catch(function () { /* keep the widget hidden */ });
    }
  }
})();
