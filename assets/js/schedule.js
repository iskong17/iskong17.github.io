/* Schedule page: conference deadlines, rebuttals, notifications and dates, and the F1 calendar, on one month view.
   Deadlines are given as a date plus the time zone the conference uses (Anywhere on Earth by default) and are
   shown as the matching Korean time. F1 session times come in UTC. Everything is shown in KST (UTC+9).
   Anything that lasts several days (a conference, a rebuttal window, an F1 weekend) is one continuous bar. */
(function () {
  'use strict';
  var el = document.getElementById('sched-data');
  if (!el) return;
  var data = JSON.parse(el.textContent);
  var KST = 9 * 3600e3, DAY = 864e5;
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  var DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  var TZ = { AoE: -12, UTC: 0, GMT: 0, PT: -8, PST: -8, PDT: -7, ET: -5, EST: -5, EDT: -4, CET: 1, CEST: 2, KST: 9 };
  var AREAS = [
    { key: 'NLP', label: 'NLP' },
    { key: 'CV', label: 'CV' },
    { key: 'Speech', label: 'Speech & Audio' },
    { key: 'MLAI', label: 'ML & AI' },
    { key: 'F1', label: 'F1' }
  ];
  var areaKey = function (a) { return a === 'ML' || a === 'AI' ? 'MLAI' : a; };
  var cls = function (a) { return 'area-' + a.toLowerCase(); };

  var isDate = function (s) { return typeof s === 'string' && /^\d{4}-\d{2}-\d{2}/.test(s); };
  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  };
  var kstDay = function (ms) { return new Date(ms + KST).toISOString().slice(0, 10); };
  var kstTime = function (ms) { return new Date(ms + KST).toISOString().slice(11, 16); };
  var dayMs = function (key) { return Date.parse(key + 'T00:00:00Z'); };
  var addDays = function (key, n) { return new Date(dayMs(key) + n * DAY).toISOString().slice(0, 10); };
  var fmtDay = function (key, withYear) {
    var d = new Date(key + 'T00:00:00Z');
    return MONTHS[d.getUTCMonth()].slice(0, 3) + ' ' + d.getUTCDate() + (withYear ? ', ' + d.getUTCFullYear() : '') + ' (' + DAYS[d.getUTCDay()] + ')';
  };
  var offset = function (tz) { return TZ[tz] !== undefined ? TZ[tz] : -12; };
  // The end of a day in the conference's time zone, and the start of one, as instants
  var endMs = function (date, tz) { return Date.parse(date + 'T23:59:59Z') - offset(tz) * 3600e3; };
  var startMs = function (date, tz) { return Date.parse(date + 'T00:00:00Z') - offset(tz) * 3600e3; };
  var sessionMs = function (s) { return Date.parse(s.length > 10 ? s : s + 'T00:00:00Z'); };
  var until = function (ms) {
    var d = ms - Date.now();
    if (d < 0) return '';
    var days = Math.floor(d / DAY);
    return days >= 1 ? 'D-' + days : Math.floor(d / 36e5) + 'h left';
  };

  // ---------- Events: {start, end (day keys, inclusive), area, kind, label, sub, anchor, ms} ----------
  var events = [];
  var confs = (data.conferences || []).slice();
  confs.forEach(function (c) {
    c.area = areaKey(c.area);
    c.id = 'conf-' + c.name.toLowerCase().replace(/[^a-z0-9]+/g, '') + '-' + c.year;
    var tag = c.name + ' ' + c.year, tz = c.deadline_tz;
    var one = function (kind, label, ms, sub) {
      events.push({ start: kstDay(ms), end: kstDay(ms), area: c.area, kind: kind, label: label, sub: sub, anchor: c.id, ms: ms });
    };
    if (isDate(c.abstract_deadline)) { var a = endMs(c.abstract_deadline, tz); one('deadline', tag + ' abstract', a, kstTime(a)); }
    if (isDate(c.submission_deadline)) { var s = endMs(c.submission_deadline, tz); one('deadline', tag + ' paper', s, kstTime(s)); }
    if (isDate(c.rebuttal_start)) {
      var rs = startMs(c.rebuttal_start, tz), re = isDate(c.rebuttal_end) ? endMs(c.rebuttal_end, tz) : rs;
      events.push({ start: kstDay(rs), end: kstDay(re), area: c.area, kind: 'rebuttal', label: tag + ' rebuttal', anchor: c.id, ms: rs, endMs: re });
    }
    if (isDate(c.notification)) one('notify', tag + ' decisions', startMs(c.notification, tz), '');
    if (isDate(c.start)) {
      events.push({ start: c.start, end: isDate(c.end) ? c.end : c.start, area: c.area, kind: 'conf', label: tag + (c.city && c.city !== 'TBD' ? ' · ' + c.city : ''), anchor: c.id, ms: dayMs(c.start) });
    }
  });
  var f1Seasons = [{ season: data.f1_season, races: data.f1 || [] }];
  if (data.f1_next && data.f1_next.races) f1Seasons.push(data.f1_next);
  f1Seasons.forEach(function (s) {
    s.races.forEach(function (r) {
      r.id = 'f1-' + s.season + '-' + r.round;
      var sess = r.sessions && r.sessions.length ? r.sessions : [{ key: 'Race', start: r.start }];
      var day = function (x) { return x.start.length > 10 ? kstDay(sessionMs(x.start)) : x.start.slice(0, 10); };
      var race = sess.filter(function (x) { return x.key === 'Race'; })[0] || sess[sess.length - 1];
      var raceMs = sessionMs(race.start);
      events.push({
        start: day(sess[0]), end: day(race), area: 'F1', kind: 'f1', anchor: r.id, ms: raceMs,
        label: r.name.replace(' Grand Prix', ' GP'), sub: race.start.length > 10 ? 'Race ' + kstTime(raceMs) : ''
      });
    });
  });

  // ---------- Filters ----------
  var active = []; // empty means all
  var chips = document.querySelectorAll('.sched-chip');
  var shown = function (area) { return !active.length || active.indexOf(area) >= 0; };
  chips.forEach(function (b) {
    b.addEventListener('click', function () {
      var a = b.getAttribute('data-area');
      if (a === 'all') active = [];
      else if (active.indexOf(a) >= 0) active.splice(active.indexOf(a), 1);
      else active.push(a);
      chips.forEach(function (c) {
        var k = c.getAttribute('data-area');
        c.classList.toggle('is-active', k === 'all' ? !active.length : active.indexOf(k) >= 0);
      });
      render();
    });
  });

  // ---------- Month view ----------
  var now = new Date(Date.now() + KST);
  var view = { y: now.getUTCFullYear(), m: now.getUTCMonth() };
  var today = kstDay(Date.now());
  document.querySelectorAll('.sched-nav').forEach(function (b) {
    b.addEventListener('click', function () {
      view.m += +b.getAttribute('data-step');
      if (view.m < 0) { view.m = 11; view.y--; }
      if (view.m > 11) { view.m = 0; view.y++; }
      renderMonth();
    });
  });
  document.querySelector('.sched-today').addEventListener('click', function () {
    view = { y: now.getUTCFullYear(), m: now.getUTCMonth() };
    renderMonth();
  });

  var ICON = { deadline: '⏰ ', notify: '📩 ', rebuttal: '✍️ ', f1: '🏁 ' };
  var title = function (e) {
    return e.label + ' · ' + (e.start === e.end ? fmtDay(e.start) : fmtDay(e.start) + ' – ' + fmtDay(e.end)) + (e.sub ? ' · ' + e.sub + ' KST' : '');
  };
  var chip = function (e) {
    return '<a class="sched-ev ev-' + e.kind + ' ' + cls(e.area) + '" href="#' + e.anchor + '" title="' + esc(title(e)) + '">' +
      (ICON[e.kind] || '') + esc(e.label) + (e.sub ? ' <small>' + esc(e.sub) + '</small>' : '') + '</a>';
  };

  function renderMonth() {
    document.getElementById('sched-month').textContent = MONTHS[view.m] + ' ' + view.y;
    var first = new Date(Date.UTC(view.y, view.m, 1));
    var gridStart = new Date(Date.UTC(view.y, view.m, 1 - first.getUTCDay())).toISOString().slice(0, 10);
    var monthKey = view.y + '-' + String(view.m + 1).padStart(2, '0');
    var weeks = Math.ceil((first.getUTCDay() + new Date(Date.UTC(view.y, view.m + 1, 0)).getUTCDate()) / 7);
    var list = events.filter(function (e) { return shown(e.area); });
    var html = '<div class="sched-dows">' + DAYS.map(function (d) { return '<div>' + d + '</div>'; }).join('') + '</div>';

    for (var w = 0; w < weeks; w++) {
      var ws = addDays(gridStart, w * 7), we = addDays(ws, 6);
      // Bars in this week, longest first so they take the top lanes
      var bars = list.filter(function (e) { return e.start <= we && e.end >= ws; }).sort(function (a, b) {
        return a.start.localeCompare(b.start) || b.end.localeCompare(a.end) || (a.ms || 0) - (b.ms || 0);
      });
      var lanes = []; // per lane, the last column used
      var placed = bars.map(function (e) {
        var c0 = e.start < ws ? 0 : (dayMs(e.start) - dayMs(ws)) / DAY;
        var c1 = e.end > we ? 6 : (dayMs(e.end) - dayMs(ws)) / DAY;
        var lane = 0;
        while (lanes[lane] !== undefined && lanes[lane] >= c0) lane++;
        lanes[lane] = c1;
        return { e: e, c0: c0, c1: c1, lane: lane, cutL: e.start < ws, cutR: e.end > we };
      });
      var rows = Math.max(lanes.length, 1);
      html += '<div class="sched-week" style="--rows:' + rows + '">';
      for (var d = 0; d < 7; d++) {
        var key = addDays(ws, d);
        html += '<div class="sched-day' + (key.slice(0, 7) !== monthKey ? ' is-out' : '') + (key === today ? ' is-today' : '') +
          '" style="grid-column:' + (d + 1) + '"><span class="sched-num">' + (+key.slice(8)) + '</span></div>';
      }
      placed.forEach(function (p) {
        var e = p.e;
        html += '<a class="sched-bar ev-' + e.kind + ' ' + cls(e.area) + (p.cutL ? ' cut-l' : '') + (p.cutR ? ' cut-r' : '') +
          '" style="grid-column:' + (p.c0 + 1) + ' / ' + (p.c1 + 2) + ';grid-row:' + (p.lane + 2) + '" href="#' + e.anchor + '" title="' + esc(title(e)) + '">' +
          '<span>' + (ICON[e.kind] || '') + esc(e.label) + (e.sub ? ' <small>' + esc(e.sub) + '</small>' : '') + '</span></a>';
      });
      html += '</div>';
    }
    document.getElementById('sched-grid').innerHTML = html;

    // Agenda: everything that touches this month, one line each (the readable view on a phone)
    var ms0 = monthKey + '-01', ms1 = addDays(new Date(Date.UTC(view.y, view.m + 1, 1)).toISOString().slice(0, 10), -1);
    var items = list.filter(function (e) { return e.start <= ms1 && e.end >= ms0; })
      .sort(function (a, b) { return a.start.localeCompare(b.start) || (a.ms || 0) - (b.ms || 0); });
    document.getElementById('sched-agenda').innerHTML = items.length ? items.map(function (e) {
      return '<li><b>' + fmtDay(e.start).replace(/ \(.*/, '') + (e.end !== e.start ? ' – ' + fmtDay(e.end).replace(/ \(.*/, '') : '') + '</b><span>' + chip(e) + '</span></li>';
    }).join('') : '<li class="sched-empty">Nothing this month.</li>';
  }

  // ---------- Next up ----------
  function renderNext() {
    var t = Date.now();
    var KIND = { deadline: 'Deadline', rebuttal: 'Rebuttal opens', notify: 'Decisions', f1: 'Race' };
    var upcoming = events.filter(function (e) { return shown(e.area) && e.ms > t && KIND[e.kind]; })
      .sort(function (a, b) { return a.ms - b.ms; }).slice(0, 4);
    document.getElementById('sched-next').innerHTML = upcoming.map(function (e) {
      return '<a class="sched-next-item ' + cls(e.area) + '" href="#' + e.anchor + '"><small>' + KIND[e.kind] + '</small><b>' + esc(e.label) + '</b><span>' +
        fmtDay(kstDay(e.ms)) + (e.kind === 'notify' ? '' : ' ' + kstTime(e.ms) + ' KST') + '</span><em>' + until(e.ms) + '</em></a>';
    }).join('');
  }

  // ---------- Conference cards ----------
  var tbd = '<span class="tbd">TBD</span>';
  var kstAt = function (ms, withYear) { return fmtDay(kstDay(ms), withYear) + ' <b>' + kstTime(ms) + '</b>'; };
  var plain = function (s) { return s.replace(/<[^>]+>/g, ''); };
  function renderConfs() {
    var t = Date.now();
    var html = AREAS.filter(function (a) { return a.key !== 'F1' && shown(a.key); }).map(function (area) {
      var rows = confs.filter(function (c) { return c.area === area.key; }).sort(function (a, b) {
        return (a.submission_deadline || 'z').localeCompare(b.submission_deadline || 'z') || a.name.localeCompare(b.name);
      });
      if (!rows.length) return '';
      return '<div class="sched-area ' + cls(area.key) + '"><h3>' + area.label + '</h3>' + rows.map(function (c) {
        var tz = c.deadline_tz;
        var dl = isDate(c.submission_deadline) ? endMs(c.submission_deadline, tz) : null;
        var over = isDate(c.end) && Date.parse(c.end + 'T23:59:59Z') < t;
        var place = [c.city, c.country].filter(function (x) { return x && x !== 'TBD'; }).join(', ');
        var row = function (k, v) { return '<div><dt>' + k + '</dt><dd>' + v + '</dd></div>'; };
        var reb;
        if (c.rebuttal_start === null && c.rebuttal_end === null) reb = '<span class="none">No rebuttal</span>';
        else if (isDate(c.rebuttal_start)) {
          var rs = startMs(c.rebuttal_start, tz), re = isDate(c.rebuttal_end) ? endMs(c.rebuttal_end, tz) : null;
          reb = fmtDay(kstDay(rs), true) + ' – ' + (re ? kstAt(re) : tbd) + ' <small>KST</small>' +
            (rs > t ? ' <em>' + until(rs) + '</em>' : re && re > t ? ' <small class="live">open now</small>' : '');
        } else reb = tbd;
        var note = isDate(c.notification) ? fmtDay(kstDay(startMs(c.notification, tz)), true) + (startMs(c.notification, tz) > t ? ' <em>' + until(startMs(c.notification, tz)) + '</em>' : '') : tbd;
        return '<article class="sched-conf' + (over ? ' is-past' : '') + '" id="' + c.id + '">' +
          '<header><b>' + esc(c.name) + ' ' + c.year + '</b>' + (c.url ? '<a href="' + esc(c.url) + '" target="_blank" rel="noopener">site ↗</a>' : '') + '</header><dl>' +
          row('Submission', (dl ? kstAt(dl, true) + ' <small>KST</small>' + (dl > t ? ' <em>' + until(dl) + '</em>' : ' <small class="done">closed</small>') : tbd) +
            (isDate(c.abstract_deadline) ? '<br><small>Abstract ' + plain(kstAt(endMs(c.abstract_deadline, tz), true)) + ' KST</small>' : '')) +
          row('Rebuttal', reb) +
          row('Notification', note) +
          row('Conference', isDate(c.start) ? fmtDay(c.start, true) + (isDate(c.end) && c.end !== c.start ? ' – ' + fmtDay(c.end, true) : '') : tbd) +
          row('Location', (place ? esc(place) : tbd) + (c.venue ? '<br><small>' + esc(c.venue) + '</small>' : '')) +
          '</dl>' + (c.note ? '<p>' + esc(c.note) + '</p>' : '') + '</article>';
      }).join('') + '</div>';
    }).join('');
    document.getElementById('sched-confs').innerHTML = html || '<p class="sched-empty">No conferences selected.</p>';
    document.getElementById('sched-confs').previousElementSibling.hidden = !html;
  }

  // ---------- F1 table ----------
  function sessCell(r, keys) {
    var s = (r.sessions || []).filter(function (x) { return keys.indexOf(x.key) >= 0; })[0];
    if (!s) return r.times_tbd ? '<td>' + tbd + '</td>' : '<td class="none">–</td>';
    if (s.start.length <= 10) return '<td>' + fmtDay(s.start) + '<br>' + tbd + '</td>';
    var ms = sessionMs(s.start);
    return '<td>' + fmtDay(kstDay(ms)) + '<br><b>' + kstTime(ms) + '</b></td>';
  }
  function renderF1() {
    var box = document.getElementById('sched-f1'), head = document.getElementById('sched-f1-title');
    var on = shown('F1');
    box.hidden = head.hidden = !on;
    if (!on) return;
    var t = Date.now();
    box.innerHTML = f1Seasons.map(function (s) {
      var next = s.races.filter(function (r) { return sessionMs(r.start) + 3 * 3600e3 > t; })[0];
      return '<h3>' + esc(s.season) + ' season</h3><div class="sched-table-wrap"><table class="sched-table">' +
        '<thead><tr><th>Rd</th><th>Grand Prix</th><th>FP1</th><th>FP2 / SQ</th><th>FP3 / Sprint</th><th>Qualifying</th><th>Race</th></tr></thead><tbody>' +
        s.races.map(function (r) {
          var past = sessionMs(r.start) + 3 * 3600e3 < t;
          var link = data.circuits && data.circuits[r.circuit_id];
          return '<tr id="' + r.id + '" class="' + (past ? 'is-past' : '') + (r === next ? ' is-next' : '') + '">' +
            '<td class="rd">' + r.round + '</td>' +
            '<td class="gp"><b>' + esc(r.name) + '</b>' + (r.sprint ? ' <span class="sprint">Sprint</span>' : '') +
            '<small>' + (link ? '<a href="' + link + '">' + esc(r.circuit) + '</a>' : esc(r.circuit)) + ' · ' + esc(r.locality) + ', ' + esc(r.country) + '</small></td>' +
            sessCell(r, ['FP1']) + sessCell(r, r.sprint ? ['SQ'] : ['FP2']) + sessCell(r, r.sprint ? ['Sprint'] : ['FP3']) + sessCell(r, ['Quali']) + sessCell(r, ['Race']) + '</tr>';
        }).join('') + '</tbody></table></div>';
    }).join('') + (data.f1_next ? '' : '<p class="sched-note">The ' + (+data.f1_season + 1) + ' calendar: ' + tbd + '</p>');
  }

  function render() { renderNext(); renderMonth(); renderConfs(); renderF1(); }
  render();
})();
