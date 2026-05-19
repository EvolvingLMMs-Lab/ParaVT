/* ============================================================
   WorldReasonBench project page
   - Reveal-on-scroll
   - Stat counters
   - Sortable + filterable Leaderboard
   - Qualitative dim tabs (placeholder videos)
   - BibTeX copy
   - Sticky nav scroll state
   ============================================================ */

(function () {
  'use strict';

  /* ------------------------------------------------------------
   * 1. Sticky nav scroll state
   * ------------------------------------------------------------ */
  const nav = document.getElementById('topnav');
  const onScroll = () => {
    if (window.scrollY > 8) nav.classList.add('scrolled');
    else nav.classList.remove('scrolled');
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ------------------------------------------------------------
   * 2. Reveal-on-scroll
   * ------------------------------------------------------------ */
  const revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('visible');
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -10% 0px' }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add('visible'));
  }

  /* ------------------------------------------------------------
   * 3. Animated stat counters
   * ------------------------------------------------------------ */
  const counters = document.querySelectorAll('.stat-num[data-target]');
  const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);
  const animateCounter = (el) => {
    const target = parseInt(el.getAttribute('data-target'), 10);
    const suffix = el.getAttribute('data-suffix') || '';
    const duration = 1400;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const v = Math.floor(easeOutQuart(t) * target);
      el.textContent = v.toLocaleString() + (t === 1 ? suffix : '');
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  if ('IntersectionObserver' in window) {
    const cio = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            animateCounter(e.target);
            cio.unobserve(e.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    counters.forEach((c) => cio.observe(c));
  } else {
    counters.forEach((c) => animateCounter(c));
  }

  /* ------------------------------------------------------------
   * 4. Leaderboard
   * ------------------------------------------------------------ */
  const lbState = {
    metric: 'score_pr',     // 'score_pr' or 'sv'
    family: 'all',          // 'all' / 'closed' / 'open'
    sortKey: 'overall',     // 'overall' / 'wk' / 'hc' / 'lr' / 'ib' / 'model' / 'family' / 'rank'
    sortDir: 'desc',        // 'asc' or 'desc'
    search: '',
    models: []
  };

  const tbody = document.getElementById('lb-tbody');
  const table = document.getElementById('lb-table');

  const dimColumns = [
    { key: 'overall', label: 'Overall' },
    { key: 'wk',      label: 'World Knowledge' },
    { key: 'hc',      label: 'Human-Centric' },
    { key: 'lr',      label: 'Logic Reasoning' },
    { key: 'ib',      label: 'Information-Based' }
  ];

  const fmt = (v) => (typeof v === 'number' ? v.toFixed(1) : '--');

  function getValue(model, key) {
    if (key === 'model') return model.name.toLowerCase();
    if (key === 'family') return model.family;
    if (key === 'rank') return model._rank;
    return model[lbState.metric][key];
  }

  function rankDescBy(arr, key) {
    return [...arr].sort((a, b) => {
      const va = getValue(a, key);
      const vb = getValue(b, key);
      if (typeof va === 'string') return va.localeCompare(vb);
      return vb - va;
    });
  }

  function computeBestPerColumn(models) {
    // compute first / second-best across the CURRENTLY VISIBLE models for each dim
    const best = {};
    dimColumns.forEach((c) => {
      const sorted = rankDescBy(models, c.key);
      best[c.key] = {
        first: sorted[0] ? getValue(sorted[0], c.key) : null,
        second: sorted[1] ? getValue(sorted[1], c.key) : null
      };
    });
    return best;
  }

  function applyFilters(allModels) {
    return allModels.filter((m) => {
      if (lbState.family !== 'all' && m.family !== lbState.family) return false;
      if (lbState.search && !m.name.toLowerCase().includes(lbState.search)) return false;
      return true;
    });
  }

  function applySort(filtered) {
    const sorted = [...filtered].sort((a, b) => {
      const va = getValue(a, lbState.sortKey);
      const vb = getValue(b, lbState.sortKey);
      if (typeof va === 'string') {
        return lbState.sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return lbState.sortDir === 'asc' ? va - vb : vb - va;
    });
    return sorted;
  }

  function renderTable() {
    // 1. assign overall rank from the FULL set (so a hidden filter doesn't change
    //    the rank number people see)
    const rankedAll = rankDescBy(lbState.models, 'overall');
    rankedAll.forEach((m, i) => { m._rank = i + 1; });

    // 2. filter -> sort
    const filtered = applyFilters(lbState.models);
    const best = computeBestPerColumn(filtered);
    const sorted = applySort(filtered);

    // 3. render rows
    tbody.innerHTML = '';
    sorted.forEach((m, i) => {
      const tr = document.createElement('tr');
      tr.classList.add('lb-row-anim');
      tr.style.animationDelay = (i * 18) + 'ms';

      const familyBadge = m.family === 'closed'
        ? '<span class="fam-badge fam-closed"><span class="fam-dot"></span>Closed</span>'
        : '<span class="fam-badge fam-open"><span class="fam-dot"></span>Open</span>';

      let html =
        '<td class="col-rank">' + m._rank + '</td>' +
        '<td class="col-model">' + escapeHtml(m.name) + '</td>' +
        '<td class="col-family">' + familyBadge + '</td>';

      dimColumns.forEach((c) => {
        const v = getValue(m, c.key);
        let cls = 'col-num';
        if (v === best[c.key].first && v !== null)        cls += ' is-best';
        else if (v === best[c.key].second && v !== null)  cls += ' is-second';
        html += '<td class="' + cls + '">' + fmt(v) + '</td>';
      });

      tr.innerHTML = html;
      tbody.appendChild(tr);
    });

    // 4. highlight active sort column header
    table.querySelectorAll('thead th').forEach((th) => {
      th.classList.remove('is-active', 'asc');
      if (th.getAttribute('data-sort') === lbState.sortKey) {
        th.classList.add('is-active');
        if (lbState.sortDir === 'asc') th.classList.add('asc');
      }
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  // --- bind controls ---
  document.querySelectorAll('.lb-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lb-tab').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      lbState.metric = btn.getAttribute('data-metric');
      renderTable();
    });
  });

  document.querySelectorAll('.lb-chip').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lb-chip').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      lbState.family = btn.getAttribute('data-family');
      renderTable();
    });
  });

  table.querySelectorAll('thead th').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.getAttribute('data-sort');
      if (lbState.sortKey === key) {
        lbState.sortDir = lbState.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        lbState.sortKey = key;
        lbState.sortDir = (key === 'model' || key === 'family') ? 'asc' : 'desc';
      }
      renderTable();
    });
  });

  const searchInput = document.getElementById('lb-search-input');
  searchInput.addEventListener('input', (e) => {
    lbState.search = e.target.value.trim().toLowerCase();
    renderTable();
  });

  // --- load data ---
  fetch('data/leaderboard.json', { cache: 'no-cache' })
    .then((r) => r.json())
    .then((data) => {
      lbState.models = data.models;
      renderTable();
    })
    .catch((err) => {
      console.error('Leaderboard load failed', err);
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:24px;color:#777e9a;">Could not load leaderboard data.</td></tr>';
    });

  /* ------------------------------------------------------------
   * 5. Qualitative dimension tabs (placeholder videos)
   * ------------------------------------------------------------ */
  const videoData = {
    wk: [
      { title: 'A balloon released indoors rises and rests against the ceiling.',  cat: 'World Knowledge', src: 'data/video/WorldKnowledge/wk_1.mp4', model: '----' },
      { title: 'Ice cubes drop into hot water and gradually melt away.',            cat: 'World Knowledge', src: 'data/video/WorldKnowledge/wk_2.mp4', model: '----' },
      { title: 'A spinning top slowly loses momentum and falls over.',              cat: 'World Knowledge', src: 'data/video/WorldKnowledge/wk_3.mp4', model: '----' }
    ],
    hc: [
      { title: 'Two people shake hands and then exchange a written document.',     cat: 'Human-Centric', src: 'data/video/HumanCentric/hc_1.mp4', model: '----' },
      { title: 'A pianist plays a chord; the keys depress in synchronised order.', cat: 'Human-Centric', src: 'data/video/HumanCentric/hc_2.mp4', model: '----' },
      { title: 'A child reaches for a cup; the parent catches it before it tips.', cat: 'Human-Centric', src: 'data/video/HumanCentric/hc_3.mp4', model: '----' }
    ],
    lr: [
      { title: 'Three numbered cups, marble under one, after a swap and reveal.',   cat: 'Logic Reasoning', src: 'data/video/LogicReasoning/lr_1.mp4', model: '----' },
      { title: 'A maze runner takes the shortest valid path to the exit.',          cat: 'Logic Reasoning', src: 'data/video/LogicReasoning/lr_2.mp4', model: '----' },
      { title: 'A scale balances after replacing one side with equal mass.',        cat: 'Logic Reasoning', src: 'data/video/LogicReasoning/lr_3.mp4', model: '----' }
    ],
    ib: [
      { title: 'A whiteboard equation is partly erased then rewritten correctly.',  cat: 'Information-Based', src: 'data/video/InformationBased/ib_1.mp4', model: '----' },
      { title: 'A digital clock counts forward by exactly five seconds.',           cat: 'Information-Based', src: 'data/video/InformationBased/ib_2.mp4', model: '----' },
      { title: 'A book page turns and reveals the same paragraph re-typeset.',      cat: 'Information-Based', src: 'data/video/InformationBased/ib_3.mp4', model: '----' }
    ]
  };

  const dimToCategory = {
    wk: 'WorldKnowledge',
    hc: 'HumanCentric',
    lr: 'LogicReasoning',
    ib: 'InformationBased'
  };

  let activeVideoDim = 'wk';
  const loadedVideoInfo = new Set();
  const expandedVideoDims = new Set();

  const grid = document.getElementById('video-grid');
  const videoExpandBtn = document.getElementById('video-expand-btn');
  const videosPerDimension = 3;
  const collapsedVideoCount = 3;
  const promptToggleLabel = (expanded) => expanded ? 'Collapse prompt' : 'Show full prompt';

  function syncPromptToggles() {
    grid.querySelectorAll('.vm-prompt-wrap').forEach((wrap) => {
      const prompt = wrap.querySelector('.vm-title');
      const toggle = wrap.querySelector('.vm-prompt-toggle');
      if (!prompt || !toggle) return;

      const isExpanded = prompt.classList.contains('expanded');
      if (isExpanded) {
        toggle.hidden = false;
        return;
      }

      toggle.hidden = prompt.scrollHeight <= prompt.clientHeight + 1;
    });
  }

  function renderVideos(dim) {
    activeVideoDim = dim;
    const sourceItems = videoData[dim] || [];
    const categoryLabel = sourceItems[0] ? sourceItems[0].cat : 'Video';
    const allItems = Array.from({ length: videosPerDimension }, (_, index) => (
      sourceItems[index] || {
        title: 'Coming soon',
        cat: categoryLabel,
        src: '',
        model: '----'
      }
    ));
    const isExpanded = expandedVideoDims.has(dim);
    const items = isExpanded ? allItems : allItems.slice(0, collapsedVideoCount);

    if (videoExpandBtn) {
      videoExpandBtn.hidden = allItems.length <= collapsedVideoCount;
      videoExpandBtn.setAttribute('aria-expanded', String(isExpanded));
      videoExpandBtn.innerHTML = isExpanded
        ? '<i class="fa-solid fa-chevron-up"></i> Show fewer videos'
        : '<i class="fa-solid fa-chevron-down"></i> Show more videos';
    }

    grid.innerHTML = items.map((v, i) => `
      <div class="video-item" style="animation: rowFadeIn 0.45s ${i * 90}ms ease both;">
        <div class="video-poster ${v.src && v.title !== '---' ? 'has-video' : ''}">
          ${v.src && v.title !== '---' ? `
            <video controls autoplay muted loop playsinline preload="metadata">
              <source src="${v.src}" type="video/mp4">
            </video>
            <div class="video-model-badge">${escapeHtml(v.model || '----')}</div>
          ` : `
            <div class="play-icon"><i class="fa-solid fa-play"></i></div>
            <span class="placeholder-tag">Coming soon</span>
          `}
        </div>
        <div class="video-meta">
          <div class="vm-cat">${v.cat}</div>
          <div class="vm-prompt-wrap">
            <div class="vm-title">${escapeHtml(v.title)}</div>
            <button class="vm-prompt-toggle" type="button" aria-expanded="false" aria-label="${promptToggleLabel(false)}" title="${promptToggleLabel(false)}" hidden>
              <i class="fa-solid fa-ellipsis"></i>
            </button>
          </div>
        </div>
      </div>
    `).join('');
    requestAnimationFrame(syncPromptToggles);
  }

  function applyVideoInfo(dim, categoryData) {
    const items = videoData[dim] || [];
    const category = dimToCategory[dim];
    if (!category || !categoryData || !Array.isArray(categoryData.videos)) return;

    categoryData.videos.forEach((video, index) => {
      const isPlaceholder = video.video_name === '---';
      const videoName = isPlaceholder ? `${dim}_${index + 1}.mp4` : video.video_name;
      items[index] = {
        title: video.prompt || '---',
        cat: items[index] ? items[index].cat : categoryData.category,
        src: `data/video/${category}/${videoName}`,
        model: video.model || '---'
      };
    });
  }

  function loadVideoInfo(dim) {
    const category = dimToCategory[dim];
    if (!category || loadedVideoInfo.has(dim)) return;

    fetch(`data/${category}.json`, { cache: 'no-cache' })
      .then((r) => {
        if (!r.ok) throw new Error(`Could not load data/${category}.json`);
        return r.json();
      })
      .then((data) => {
        applyVideoInfo(dim, data);
        loadedVideoInfo.add(dim);
        if (activeVideoDim === dim) renderVideos(dim);
      })
      .catch((err) => {
        console.error('Video prompt load failed', err);
      });
  }

  grid.addEventListener('click', (e) => {
    const toggle = e.target.closest('.vm-prompt-toggle');
    if (!toggle) return;

    const wrap = toggle.closest('.vm-prompt-wrap');
    const prompt = wrap && wrap.querySelector('.vm-title');
    if (!prompt) return;

    const expanded = prompt.classList.toggle('expanded');
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', promptToggleLabel(expanded));
    toggle.setAttribute('title', promptToggleLabel(expanded));
    toggle.innerHTML = expanded
      ? '<i class="fa-solid fa-chevron-up"></i>'
      : '<i class="fa-solid fa-ellipsis"></i>';
  });
  window.addEventListener('resize', syncPromptToggles);
  if (videoExpandBtn) {
    videoExpandBtn.addEventListener('click', () => {
      if (expandedVideoDims.has(activeVideoDim)) {
        expandedVideoDims.delete(activeVideoDim);
      } else {
        expandedVideoDims.add(activeVideoDim);
      }
      renderVideos(activeVideoDim);
    });
  }
  document.querySelectorAll('.dim-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.dim-tab').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const dim = btn.getAttribute('data-dim');
      expandedVideoDims.delete(dim);
      renderVideos(dim);
      loadVideoInfo(dim);
    });
  });
  renderVideos(activeVideoDim);
  loadVideoInfo(activeVideoDim);

  /* ------------------------------------------------------------
   * 6. BibTeX copy
   * ------------------------------------------------------------ */
  const copyBtn = document.getElementById('copy-bibtex');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const text = document.getElementById('bibtex-content').textContent;
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.classList.add('copied');
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
        }, 1600);
      } catch (e) {
        console.error('Copy failed', e);
      }
    });
  }

})();
