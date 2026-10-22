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
    // Study and Gallery filter by topic tab; Paper Review filters by subfield chip and venue chip.
    var paperConf = 'all';
    function showTopic(panel, key) {
      var bySub = !!panel.querySelector('.subfield-chip');
      var attr = bySub ? 'data-subfield' : 'data-topic';
      var tabs = panel.querySelectorAll(bySub ? '.subfield-chip' : '.topic-tabs [data-topic]');
      if (!tabs.length) return;
      if (!panel.querySelector('[' + attr + '="' + key + '"]')) key = 'all';
      // Study's All view is a bar per topic; the flat grid shows only for one topic.
      var bars = panel.querySelector('.topic-bars');
      if (bars) bars.hidden = key !== 'all';
      var cards = panel.querySelectorAll('.sq-grid > .sq-card, .paper-row, .study-group .lecture-row');
      var any = !!bars && key === 'all';
      cards.forEach(function (c) {
        var show = key === 'all' || (bySub
          ? (' ' + (c.getAttribute('data-subfields') || '') + ' ').indexOf(' ' + key + ' ') !== -1
          : c.getAttribute('data-topic') === key);
        if (bySub && paperConf !== 'all' && c.getAttribute('data-conf') !== paperConf) show = false;
        c.hidden = !show;
        if (show) any = true;
      });
      // Study groups (notes, Discussions) hide on All and when none of their cards match.
      panel.querySelectorAll('.study-group').forEach(function (g) {
        g.hidden = (!!bars && key === 'all') || !g.querySelector('.sq-card:not([hidden]), .lecture-row:not([hidden])');
      });
      activate(tabs, attr, key);
      if (bySub) activate(panel.querySelectorAll('.conf-chip'), 'data-conf', paperConf);
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
      b.addEventListener('click', function () {
        if (b.hasAttribute('data-conf-goto')) paperConf = b.getAttribute('data-conf-goto');
        else if (b.getAttribute('data-goto').indexOf('paper-review') === 0) paperConf = 'all';
        go('#' + b.getAttribute('data-goto'));
        window.scrollTo(0, blogTabs.offsetTop - 80);
      });
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
    // Venue chips and sort buttons on the Paper Review panel. The venue filter combines with the topic chip in the URL.
    var currentSub = function () { return (location.hash.replace(/^#/, '').split('/')[1]) || 'all'; };
    document.querySelectorAll('.conf-chip').forEach(function (b) {
      b.addEventListener('click', function () {
        paperConf = b.getAttribute('data-conf');
        showTopic(b.closest('.blog-panel'), currentSub());
      });
    });
    var paperList = document.querySelector('.paper-list');
    var sortPapers = function (by) {
      if (!paperList) return;
      var rows = Array.prototype.slice.call(paperList.querySelectorAll('.paper-row'));
      var title = function (r) { return (r.getAttribute('data-title') || '').toLowerCase(); };
      rows.sort(function (a, b) {
        if (by === 'title') return title(a).localeCompare(title(b));
        var dy = (+b.getAttribute('data-year') || 0) - (+a.getAttribute('data-year') || 0);
        return dy || (a.getAttribute('data-conf') || '').localeCompare(b.getAttribute('data-conf') || '') || title(a).localeCompare(title(b));
      });
      rows.forEach(function (r) { paperList.appendChild(r); });
      activate(document.querySelectorAll('.sort-btn'), 'data-sort', by);
    };
    document.querySelectorAll('.sort-btn').forEach(function (b) {
      b.addEventListener('click', function () { sortPapers(b.getAttribute('data-sort')); });
    });
    sortPapers('year');
    document.querySelectorAll('.subfield-chip').forEach(function (b) {
      var panel = b.closest('.blog-panel').getAttribute('data-panel');
      b.addEventListener('click', function () {
        var sf = b.getAttribute('data-subfield');
        go('#' + panel + (sf === 'all' ? '' : '/' + sf));
      });
    });
    window.addEventListener('hashchange', function () { route(location.hash); });
    route(location.hash);

    // Search: filters every post by title, description, key concepts, authors and so on.
    // While a query is typed the tabs and panels step aside for a single result list.
    var searchBox = document.getElementById('blog-search');
    var results = document.getElementById('search-results');
    var dataEl = document.getElementById('search-data');
    if (searchBox && results && dataEl) {
      var index = [];
      try { index = JSON.parse(dataEl.textContent); } catch (e) { index = []; }
      index.forEach(function (p) { p.h = [p.t, p.f, p.c, p.s, p.x, p.e].join(' ').toLowerCase(); });
      var list = document.querySelector('.blog-list');
      var esc = function (str) {
        return String(str || '').replace(/[&<>"]/g, function (ch) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]; });
      };
      var search = function () {
        var q = searchBox.value.trim().toLowerCase();
        list.classList.toggle('is-searching', q.length > 0);
        results.hidden = !q;
        if (!q) { results.innerHTML = ''; return; }
        var terms = q.split(/\s+/);
        var hits = index.filter(function (p) { return terms.every(function (t) { return p.h.indexOf(t) !== -1; }); });
        var html = '<p class="search-count"><b>' + hits.length + '</b> ' + (hits.length === 1 ? 'result' : 'results') + ' for “' + esc(searchBox.value.trim()) + '”</p>';
        html += hits.slice(0, 60).map(function (p) {
          return '<a class="search-hit cat-' + esc(p.k) + '" href="' + esc(p.u) + '">' +
            '<span class="search-hit-meta"><span class="search-hit-cat">' + esc(p.c) + '</span>' + (p.s ? '<span>' + esc(p.s) + '</span>' : '') + '<time>' + esc(p.d) + '</time></span>' +
            '<b class="search-hit-title">' + esc(p.t) + '</b>' +
            (p.x ? '<span class="search-hit-desc">' + esc(p.x) + '</span>' : '') + '</a>';
        }).join('');
        if (hits.length > 60) html += '<p class="search-count">Showing the first 60. Add a word to narrow it down.</p>';
        if (!hits.length) html += '<p class="empty">Nothing matches. Try another word.</p>';
        results.innerHTML = html;
      };
      searchBox.addEventListener('input', search);
      searchBox.addEventListener('keydown', function (e) { if (e.key === 'Escape') { searchBox.value = ''; search(); } });
      document.addEventListener('keydown', function (e) {
        if (e.key === '/' && document.activeElement !== searchBox && !/input|textarea/i.test(document.activeElement.tagName)) {
          e.preventDefault(); searchBox.focus();
        }
      });
    }
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

  // Post table of contents: a fixed side panel on wide screens for posts with several sections.
  // It lists the h2/h3 headings and highlights the one being read.
  var postBody = document.querySelector('.post-body');
  if (postBody && postBody.querySelectorAll('h2').length >= 3) {
    var heads = Array.prototype.slice.call(postBody.querySelectorAll('h2, h3'));
    var toc = document.createElement('nav');
    toc.className = 'post-toc';
    toc.setAttribute('aria-label', 'Contents');
    var listHtml = '<p class="post-toc-title">Contents</p><ol>';
    heads.forEach(function (h, i) {
      if (!h.id) h.id = 'section-' + (i + 1);
      var label = h.textContent.replace(/\$\$/g, '').trim();
      listHtml += '<li class="toc-' + h.tagName.toLowerCase() + '"><a href="#' + h.id + '">' +
        label.replace(/[&<>"]/g, function (ch) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]; }) + '</a></li>';
    });
    toc.innerHTML = listHtml + '</ol>';
    postBody.parentNode.appendChild(toc);
    var links = toc.querySelectorAll('a');
    var setActive = function () {
      var current = 0;
      heads.forEach(function (h, i) { if (h.getBoundingClientRect().top < 140) current = i; });
      links.forEach(function (a, i) { a.classList.toggle('is-active', i === current); });
      var active = links[current];
      if (active && toc.scrollHeight > toc.clientHeight) {
        var top = active.offsetTop - toc.clientHeight / 2;
        if (Math.abs(toc.scrollTop - top) > toc.clientHeight / 3) toc.scrollTop = top;
      }
    };
    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () { setActive(); ticking = false; });
    }, { passive: true });
    setActive();
  }
})();
