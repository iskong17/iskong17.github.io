/* Schedule page: conference deadlines and dates, and the F1 calendar, on one month view.
   Conference deadlines are Anywhere on Earth (UTC-12) unless the data says otherwise; F1 times come in UTC.
   Everything is shown in Korean time (UTC+9). */
(function () {
  'use strict';
  var el = document.getElementById('sched-data');
  if (!el) return;
  var data = JSON.parse(el.textContent);
  var KST = 9 * 3600e3;
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  var DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  var TZ = { AoE: -12, UTC: 0, PT: -8, PST: -8, PDT: -7, ET: -5, EST: -5, EDT: -4, CET: 1, CEST: 2, KST: 9 };

  var isDate = function (s) { return typeof s === 'string' && /^\d{4}-\d{2}-\d{2}/.test(s); };
  var esc = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  };
  // A calendar day key from a date string, no time zone shift
  var dayKey = function (s) { return s.slice(0, 10); };
  // The KST calendar day of an instant
  var kstDay = function (ms) { return new Date(ms + KST).toISOString().slice(0, 10); };
  var kstTime = function (ms) { return new Date(ms + KST).toISOString().slice(11, 16); };
  var fmtDay = function (key, withYear) {
    var d = new Date(key + 'T00:00:00Z');
    return MONTHS[d.getUTCMonth()].slice(0, 3) + ' ' + d.getUTCDate() + (withYear ? ', ' + d.getUTCFullYear() : '') + ' (' + DAYS[d.getUTCDay()] + ')';
  };
  // End of the deadline day in its time zone, as an instant
  var deadlineMs = function (date, tz) {
    var off = TZ[tz] !== undefined ? TZ[tz] : -12;
    return Date.parse(date + 'T23:59:59Z') - off * 3600e3;
  };
  var sessionMs = function (s) { return Date.parse(s.length > 10 ? s : s + 'T00:00:00Z'); };
  var until = function (ms) {
    var d = ms - Date.now();
    if (d < 0) return '';
    var days = Math.floor(d / 864e5);
    return days >= 1 ? 'D-' + days : Math.floor(d / 36e5) + 'h left';
  };

  // ---------- Events ----------
  var events = []; // {day, area, kind, label, sub, ms?, anchor}
  var confs = (data.conferences || []).slice();
  confs.forEach(function (c, i) {
    c.id = 'conf-' + c.name.toLowerCase() + '-' + c.year;
    var tag = c.name + ' ' + c.year;
    [['abstract_deadline', ' abstract'], ['submission_deadline', ' deadline']].forEach(function (d) {
      if (!isDate(c[d[0]])) return;
      var ms = deadlineMs(c[d[0]], c.deadline_tz);
      events.push({ day: kstDay(ms), area: c.area, kind: 'deadline', label: tag + d[1], sub: kstTime(ms), anchor: c.id, ms: ms });
    });
    if (isDate(c.notification)) events.push({ day: dayKey(c.notification), area: c.area, kind: 'note', label: tag + ' notification', anchor: c.id });
    if (isDate(c.start)) {
      var end = isDate(c.end) ? c.end : c.start;
      for (var t = Date.parse(c.start + 'T00:00:00Z'); t <= Date.parse(end + 'T00:00:00Z'); t += 864e5) {
        var k = new Date(t).toISOString().slice(0, 10);
        events.push({ day: k, area: c.area, kind: 'conf', label: tag, sub: k === c.start ? (c.city || '') : '', anchor: c.id, first: k === c.start });
      }
    }
  });
  var f1Seasons = [{ season: data.f1_season, races: data.f1 || [] }];
  if (data.f1_next && data.f1_next.races) f1Seasons.push(data.f1_next);
  f1Seasons.forEach(function (s) {
    s.races.forEach(function (r) {
      r.id = 'f1-' + s.season + '-' + r.round;
      (r.sessions || [{ key: 'Race', start: r.start }]).forEach(function (x) {
        var ms = sessionMs(x.start), timed = x.start.length > 10;
        events.push({
          day: timed ? kstDay(ms) : dayKey(x.start), area: 'F1', kind: x.key === 'Race' ? 'race' : 'session',
          label: (x.key === 'Race' ? r.name.replace(' Grand Prix', ' GP') : x.key + ' · ' + r.locality), sub: timed ? kstTime(ms) : 'TBD', anchor: r.id, ms: ms
        });
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

  var pill = function (e) {
    return '<a class="sched-ev ev-' + e.kind + ' area-' + e.area.toLowerCase() + '" href="#' + e.anchor + '" title="' + esc(e.label + (e.sub ? ' · ' + e.sub : '')) + '">' +
      (e.kind === 'deadline' ? '⏰ ' : '') + esc(e.label) + (e.sub && e.kind !== 'conf' ? ' <small>' + esc(e.sub) + '</small>' : '') + '</a>';
  };

  function renderMonth() {
    document.getElementById('sched-month').textContent = MONTHS[view.m] + ' ' + view.y;
    var first = new Date(Date.UTC(view.y, view.m, 1));
    var start = Date.UTC(view.y, view.m, 1 - first.getUTCDay());
    var monthKey = view.y + '-' + String(view.m + 1).padStart(2, '0');
    var byDay = {};
    events.forEach(function (e) {
      if (!shown(e.area)) return;
      (byDay[e.day] = byDay[e.day] || []).push(e);
    });
    // One pill per conference per day, sessions of one weekend after the race-free days
    var html = DAYS.map(function (d) { return '<div class="sched-dow">' + d + '</div>'; }).join('');
    var weeks = Math.ceil((first.getUTCDay() + new Date(Date.UTC(view.y, view.m + 1, 0)).getUTCDate()) / 7);
    for (var i = 0; i < weeks * 7; i++) {
      var key = new Date(start + i * 864e5).toISOString().slice(0, 10);
      var list = (byDay[key] || []).sort(function (a, b) { return (a.ms || 0) - (b.ms || 0); });
      var cls = 'sched-day' + (key.slice(0, 7) !== monthKey ? ' is-out' : '') + (key === today ? ' is-today' : '') + (list.length ? ' has-ev' : '');
      var dots = list.map(function (e) { return '<i class="area-' + e.area.toLowerCase() + '"></i>'; }).join('');
      html += '<div class="' + cls + '"><span class="sched-num">' + (+key.slice(8)) + '</span>' +
        '<div class="sched-evs">' + list.map(pill).join('') + '</div><div class="sched-dots">' + dots + '</div></div>';
    }
    document.getElementById('sched-grid').innerHTML = html;

    // Agenda: this month's events as a list (the readable view on a phone)
    var days = Object.keys(byDay).filter(function (k) { return k.slice(0, 7) === monthKey; }).sort();
    document.getElementById('sched-agenda').innerHTML = days.length ? days.map(function (k) {
      var list = byDay[k].filter(function (e) { return e.kind !== 'conf' || e.first || k.slice(8) === '01'; });
      if (!list.length) return '';
      return '<li><b>' + fmtDay(k) + '</b><span>' + list.map(pill).join('') + '</span></li>';
    }).join('') : '<li class="sched-empty">Nothing this month.</li>';
  }

  // ---------- Next up ----------
  function renderNext() {
    var t = Date.now();
    var upcoming = events.filter(function (e) { return shown(e.area) && e.ms && e.ms > t && (e.kind === 'deadline' || e.kind === 'race'); })
      .sort(function (a, b) { return a.ms - b.ms; }).slice(0, 4);
    document.getElementById('sched-next').innerHTML = upcoming.map(function (e) {
      return '<a class="sched-next-item area-' + e.area.toLowerCase() + '" href="#' + e.anchor + '"><small>' + (e.kind === 'deadline' ? 'Deadline' : 'Race') + '</small><b>' + esc(e.label) + '</b><span>' +
        fmtDay(kstDay(e.ms)) + ' ' + kstTime(e.ms) + ' KST</span><em>' + until(e.ms) + '</em></a>';
    }).join('');
  }

  // ---------- Conference table ----------
  var tbd = '<span class="tbd">TBD</span>';
  // A deadline as its KST date and time
  var kstDeadline = function (s, tz) {
    var ms = deadlineMs(s, tz);
    return fmtDay(kstDay(ms), true) + ' <b>' + kstTime(ms) + '</b> <small>KST</small>';
  };
  function renderConfs() {
    var t = Date.now();
    var order = ['NLP', 'CV', 'ML', 'AI'];
    var html = order.filter(shown).map(function (area) {
      var rows = confs.filter(function (c) { return c.area === area; }).sort(function (a, b) {
        return (a.submission_deadline || 'z').localeCompare(b.submission_deadline || 'z') || a.name.localeCompare(b.name);
      });
      if (!rows.length) return '';
      return '<div class="sched-area area-' + area.toLowerCase() + '"><h3>' + area + '</h3>' + rows.map(function (c) {
        var dl = isDate(c.submission_deadline) ? deadlineMs(c.submission_deadline, c.deadline_tz) : null;
        var over = isDate(c.end) && Date.parse(c.end + 'T23:59:59Z') < t;
        var place = [c.city, c.country].filter(function (x) { return x && x !== 'TBD'; }).join(', ');
        return '<article class="sched-conf' + (over ? ' is-past' : '') + '" id="' + c.id + '">' +
          '<header><b>' + esc(c.name) + ' ' + c.year + '</b>' + (c.url ? '<a href="' + esc(c.url) + '" target="_blank" rel="noopener">site ↗</a>' : '') + '</header>' +
          '<dl>' +
          '<div><dt>Submission</dt><dd>' + (isDate(c.submission_deadline) ? kstDeadline(c.submission_deadline, c.deadline_tz) : tbd) +
          (dl ? (dl > t ? ' <em>' + until(dl) + '</em>' : ' <small class="done">closed</small>') : '') +
          (isDate(c.abstract_deadline) ? '<br><small>Abstract ' + kstDeadline(c.abstract_deadline, c.deadline_tz).replace(/<\/?(b|small)>/g, '') + '</small>' : '') + '</dd></div>' +
          '<div><dt>Conference</dt><dd>' + (isDate(c.start) ? fmtDay(c.start, true) + (isDate(c.end) && c.end !== c.start ? ' – ' + fmtDay(c.end, true) : '') : tbd) + '</dd></div>' +
          '<div><dt>Location</dt><dd>' + (place ? esc(place) : tbd) + (c.venue ? '<br><small>' + esc(c.venue) + '</small>' : '') + '</dd></div>' +
          '</dl>' + (c.note ? '<p>' + esc(c.note) + '</p>' : '') + '</article>';
      }).join('') + '</div>';
    }).join('');
    document.getElementById('sched-confs').innerHTML = html || '<p class="sched-empty">No conferences selected.</p>';
    document.getElementById('sched-confs').previousElementSibling.hidden = !html;
  }

  // ---------- F1 table ----------
  function sessCell(r, keys) {
    var s = (r.sessions || []).filter(function (x) { return keys.indexOf(x.key) >= 0; })[0];
    if (!s) return '<td class="none">–</td>';
    if (s.start.length <= 10) return '<td>' + fmtDay(s.start) + '<br>' + tbd + '</td>';
    var ms = sessionMs(s.start);
    return '<td>' + fmtDay(kstDay(ms)) + '<br><b>' + kstTime(ms) + '</b></td>';
  }
  function renderF1() {
    var box = document.getElementById('sched-f1'), title = document.getElementById('sched-f1-title');
    var on = shown('F1');
    box.hidden = title.hidden = !on;
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
    }).join('') + (data.f1_next ? '' : '<p class="sched-note">The next season\'s calendar: ' + tbd + '</p>');
  }

  function render() { renderNext(); renderMonth(); renderConfs(); renderF1(); }
  render();
})();
