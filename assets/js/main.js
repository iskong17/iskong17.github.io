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

  // Blog: filter post cards by category.
  var blogTabs = document.querySelector('.blog-list .session-tabs');
  if (blogTabs) {
    setupFilter(blogTabs, document.querySelectorAll('.blog-list .post-card'), function (card, f) {
      return f === 'all' || card.dataset.category === f;
    }, (location.hash || '#all').slice(1));
  }

  // Lightbox for publication figures.
  var lightbox = document.getElementById('lightbox');
  var lightboxImg = document.getElementById('lightbox-img');
  if (lightbox && lightboxImg) {
    document.querySelectorAll('img[data-zoom]').forEach(function (img) {
      img.addEventListener('click', function () {
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt;
        lightbox.classList.add('is-open');
      });
    });
    function close() { lightbox.classList.remove('is-open'); }
    lightbox.addEventListener('click', close);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
  }

  // Copy buttons (BibTeX on project pages).
  document.querySelectorAll('[data-copy]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var target = document.getElementById(btn.getAttribute('data-copy'));
      if (!target || !navigator.clipboard) return;
      navigator.clipboard.writeText(target.innerText).then(function () {
        var label = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(function () { btn.textContent = label; }, 1800);
      });
    });
  });
})();
