/* Dictation Scenarios: record spontaneous dictations against scenario cards.
 * No build step. State lives in IndexedDB (db "dictation-scenarios", stores "takes" and "meta").
 * See app/README.md for the export format. */
(() => {
  'use strict';

  const APP_VERSION = '1.0.0';
  const SCENARIOS_URL = '../recording/scenarios.json';
  const SAMPLE_URL = 'sample-scenarios.json';
  const DB_NAME = 'dictation-scenarios';
  const AUDIO_CONSTRAINTS = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
  };
  const AUDIO_BITRATE = 128000;
  const MIME_CANDIDATES = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4;codecs=mp4a.40.2',
    'audio/mp4',
  ];

  // ---------- IndexedDB ----------
  let dbPromise = null;
  function openDb() {
    if (!dbPromise) {
      dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => {
          const d = req.result;
          const takes = d.createObjectStore('takes', { keyPath: 'key', autoIncrement: true });
          takes.createIndex('scenario_id', 'scenario_id');
          d.createObjectStore('meta', { keyPath: 'k' });
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    return dbPromise;
  }
  async function run(storeName, mode, fn) {
    const d = await openDb();
    return new Promise((resolve, reject) => {
      let tx;
      try {
        tx = d.transaction(storeName, mode, { durability: 'strict' });
      } catch (e) {
        tx = d.transaction(storeName, mode);
      }
      let out;
      const r = fn(tx.objectStore(storeName));
      if (r) r.onsuccess = () => { out = r.result; };
      tx.oncomplete = () => resolve(out);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error('transaction aborted'));
    });
  }
  const dbAllTakes = () => run('takes', 'readonly', (s) => s.getAll());
  const dbPutTake = (t) => run('takes', 'readwrite', (s) => s.put(t));
  const dbDeleteTake = (key) => run('takes', 'readwrite', (s) => s.delete(key));
  const dbClearTakes = () => run('takes', 'readwrite', (s) => s.clear());
  const dbGetMeta = async (k) => { const r = await run('meta', 'readonly', (s) => s.get(k)); return r ? r.v : undefined; };
  const dbSetMeta = (k, v) => run('meta', 'readwrite', (s) => s.put({ k, v }));
  const dbClearMeta = () => run('meta', 'readwrite', (s) => s.clear());

  // ---------- State ----------
  const S = {
    data: null,
    raw: '',
    source: '',
    isSample: false,
    scenarios: [],
    idx: 0,
    view: 'card',
    takes: [],
    skipped: new Set(),
    rec: null,
    lastExportAt: null,
    zip: null, // { blob, name, url, takeKeys }
    urls: [],
    persisted: null,
    confirmClear: false,
  };

  // ---------- Helpers ----------
  const $ = (id) => document.getElementById(id);
  // replaceChildren() would render null as the text "null"; drop empties first.
  const setKids = (root, ...kids) => root.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
        else if (k in el && typeof v !== 'string') el[k] = v;
        else el.setAttribute(k, v === true ? '' : v);
      }
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  function fmt(sec) {
    if (sec == null || !isFinite(sec)) return '?';
    const s = Math.max(0, Math.round(sec));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }
  function fmtBytes(n) {
    if (!n) return '0 KB';
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }
  function fmtWhen(iso) {
    if (!iso) return 'never';
    try { return new Date(iso).toLocaleString(); } catch (e) { return iso; }
  }
  function safeId(id) { return String(id).replace(/[^A-Za-z0-9_-]/g, '_'); }
  function extFor(mime) {
    const m = (mime || '').toLowerCase();
    if (m.includes('webm')) return 'webm';
    if (m.includes('ogg')) return 'ogg';
    if (m.includes('mp4') || m.includes('m4a')) return 'm4a';
    if (m.includes('aac')) return 'aac';
    if (m.includes('wav')) return 'wav';
    return 'bin';
  }
  function takeFileName(t) { return `${safeId(t.scenario_id)}_t${t.take}.${t.ext || extFor(t.mimeType)}`; }
  function range(sc) {
    const ts = Array.isArray(sc && sc.target_seconds) ? sc.target_seconds : [];
    const min = typeof ts[0] === 'number' ? ts[0] : null;
    const max = typeof ts[1] === 'number' ? ts[1] : null;
    return { min, max };
  }
  function rangeText(sc) {
    const { min, max } = range(sc);
    if (min != null && max != null) return `${fmt(min)}–${fmt(max)}`;
    if (min != null) return `${fmt(min)}+`;
    if (max != null) return `up to ${fmt(max)}`;
    return '';
  }
  function rangeClass(sc, sec) {
    const { min, max } = range(sc);
    if (min != null && sec < min) return 't-under';
    if (max != null && sec > max) return 't-over';
    if (min == null && max == null) return '';
    return 't-in';
  }
  let toastTimer = null;
  function toast(msg, ms = 3500) {
    const el = $('toast');
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }
  function revokeUrls() { S.urls.forEach((u) => URL.revokeObjectURL(u)); S.urls = []; }
  function blobUrl(blob) { const u = URL.createObjectURL(blob); S.urls.push(u); return u; }

  const current = () => S.scenarios[S.idx];
  const takesFor = (id) => S.takes.filter((t) => t.scenario_id === id).sort((a, b) => a.take - b.take);
  const keptFor = (id) => takesFor(id).filter((t) => t.status === 'kept');
  const pendingFor = (id) => takesFor(id).filter((t) => t.status === 'pending');
  const isDone = (id) => keptFor(id).length > 0;
  function primaryKey(id) { const k = keptFor(id); return k.length ? k[k.length - 1].key : null; }
  const doneCount = () => S.scenarios.filter((s) => isDone(s.id)).length;

  // ---------- Loading ----------
  async function loadScenarios() {
    let res;
    try {
      res = await fetch(SCENARIOS_URL, { cache: 'no-cache' });
    } catch (e) {
      throw new Error(`Could not fetch ${SCENARIOS_URL} (offline and not cached yet?)`);
    }
    if (res.ok) return { raw: await res.text(), source: 'recording/scenarios.json', isSample: false };
    if (res.status !== 404) throw new Error(`${SCENARIOS_URL} returned HTTP ${res.status}`);
    const s = await fetch(SAMPLE_URL, { cache: 'no-cache' });
    if (!s.ok) throw new Error(`recording/scenarios.json is missing and the sample returned HTTP ${s.status}`);
    return { raw: await s.text(), source: 'app/sample-scenarios.json', isSample: true };
  }

  async function init() {
    bindStatic();
    try {
      const { raw, source, isSample } = await loadScenarios();
      const data = JSON.parse(raw);
      if (!data || !Array.isArray(data.scenarios) || !data.scenarios.length) throw new Error('scenarios.json has no "scenarios" array');
      S.data = data; S.raw = raw; S.source = source; S.isSample = isSample;
      S.scenarios = data.scenarios.filter((s) => s && s.id != null);
    } catch (e) {
      showBanner(`Could not load scenarios: ${e.message}`, true);
      $('view-card').replaceChildren(h('p', { class: 'loading' }, 'No scenarios loaded. Reload when online.'));
      updateBar();
      return;
    }
    if (S.isSample) showBanner('SAMPLE scenarios: recording/scenarios.json was not found (404). Do not record the real set against these.');

    try {
      S.takes = await dbAllTakes();
      const skipped = await dbGetMeta('skipped');
      if (Array.isArray(skipped)) S.skipped = new Set(skipped);
      S.lastExportAt = (await dbGetMeta('lastExportAt')) || null;
      const pos = await dbGetMeta('position');
      const i = S.scenarios.findIndex((s) => s.id === pos);
      if (i >= 0) S.idx = i;
    } catch (e) {
      showBanner(`IndexedDB unavailable: recordings cannot be saved (${e.message}).`, true);
    }
    requestPersist();
    render();
  }

  async function requestPersist() {
    try {
      if (navigator.storage && navigator.storage.persisted) {
        S.persisted = await navigator.storage.persisted();
        if (!S.persisted && navigator.storage.persist) S.persisted = await navigator.storage.persist();
      }
    } catch (e) { /* unsupported */ }
  }

  function showBanner(msg, isError) {
    const b = $('banner');
    b.textContent = msg;
    b.className = isError ? 'banner error' : 'banner';
    b.hidden = false;
  }

  // ---------- Static bindings ----------
  function bindStatic() {
    $('btn-card').addEventListener('click', () => setView('card'));
    $('btn-overview').addEventListener('click', () => setView('overview'));
    $('btn-export').addEventListener('click', () => setView('export'));
    $('btn-prev').addEventListener('click', () => go(S.idx - 1));
    $('btn-next').addEventListener('click', () => go(S.idx + 1));
    $('btn-rec').addEventListener('click', () => {
      if (S.rec) stopRecording(); else startRecording();
    });
    window.addEventListener('beforeunload', (e) => {
      if (S.rec) { e.preventDefault(); e.returnValue = ''; }
    });
    if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
      navigator.serviceWorker.register('sw.js').catch(() => {});
    }
  }

  function setView(v) {
    if (S.rec && v !== 'card') { toast('Stop recording first.'); return; }
    S.view = v;
    render();
    window.scrollTo(0, 0);
  }

  function go(i) {
    if (S.rec) { toast('Stop recording first.'); return; }
    if (i < 0 || i >= S.scenarios.length) return;
    S.idx = i;
    S.view = 'card';
    dbSetMeta('position', current().id).catch(() => {});
    render();
    window.scrollTo(0, 0);
  }

  function nextOpenIndex(from) {
    const n = S.scenarios.length;
    for (let step = 1; step <= n; step++) {
      const j = (from + step) % n;
      const sc = S.scenarios[j];
      if (!isDone(sc.id) && !S.skipped.has(sc.id)) return j;
    }
    for (let step = 1; step <= n; step++) {
      const j = (from + step) % n;
      if (!isDone(S.scenarios[j].id)) return j;
    }
    return Math.min(from + 1, n - 1);
  }

  async function skip() {
    const sc = current();
    S.skipped.add(sc.id);
    await dbSetMeta('skipped', [...S.skipped]).catch(() => {});
    go(nextOpenIndex(S.idx));
  }

  // ---------- Rendering ----------
  function render() {
    revokeUrls();
    for (const v of ['card', 'overview', 'export']) {
      $(`view-${v}`).hidden = S.view !== v;
      $(`btn-${v}`).setAttribute('aria-current', S.view === v ? 'true' : 'false');
    }
    $('progress').textContent = `${doneCount()}/${S.scenarios.length}`;
    if (S.view === 'card') renderCard();
    else if (S.view === 'overview') renderOverview();
    else renderExport();
    updateBar();
  }

  function renderCard() {
    const sc = current();
    const root = $('view-card');
    if (!sc) { setKids(root, h('p', { class: 'loading' }, 'No scenarios.')); return; }
    const status = isDone(sc.id)
      ? h('span', { class: 'badge done' }, `kept ×${keptFor(sc.id).length}`)
      : S.skipped.has(sc.id) ? h('span', { class: 'badge skipped' }, 'skipped') : null;
    const details = Array.isArray(sc.details) ? sc.details : [];
    const delivery = Array.isArray(sc.delivery) ? sc.delivery : [];
    const targets = Array.isArray(sc.targets) ? sc.targets : [];
    const rt = rangeText(sc);

    const card = h('article', { class: 'card' },
      h('div', { class: 'card-head' },
        h('span', { class: 'sid' }, `${sc.id}`),
        h('span', { class: 'muted' }, `${S.idx + 1} of ${S.scenarios.length}`),
        sc.length ? h('span', { class: `badge ${sc.length}` }, sc.length) : null,
        sc.use_case ? h('span', { class: 'badge' }, String(sc.use_case).replace(/_/g, ' ')) : null,
        status,
        h('span', { class: 'spacer' }),
        isDone(sc.id) ? null : h('button', { type: 'button', class: 'link-btn', onclick: skip }, 'Skip for now'),
      ),
      h('h2', null, sc.title || '(untitled)'),
      rt ? h('p', { class: 'target' }, `Aim for ${rt}`) : null,
      sc.situation ? h('p', { class: 'situation' }, sc.situation) : null,
      details.length ? h('h3', null, 'Mention') : null,
      details.length ? h('ul', { class: 'checklist' }, details.map((d) =>
        h('li', null, h('label', null, h('input', { type: 'checkbox' }), h('span', null, d))))) : null,
      delivery.length ? h('h3', null, 'How to say it') : null,
      delivery.length ? h('ul', { class: 'cues' }, delivery.map((d) => h('li', null, d))) : null,
      targets.length ? h('details', { class: 'why' },
        h('summary', null, 'why (for the researcher)'),
        h('div', { class: 'tags' }, targets.map((t) => h('span', { class: 'tag' }, t)))) : null,
    );

    const cast = Array.isArray(S.data.cast) && S.data.cast.length
      ? h('details', { class: 'cast' },
        h('summary', null, `Cast (${S.data.cast.length})`),
        h('ul', null, S.data.cast.map((c) => h('li', null, h('b', null, c.name || '?'), c.role ? ` — ${c.role}` : ''))))
      : null;

    setKids(root, card, renderTakes(sc), cast);
  }

  function renderTakes(sc) {
    const wrap = h('section', { class: 'takes', 'aria-label': 'Takes' });
    for (const t of pendingFor(sc.id)) {
      wrap.append(h('div', { class: 'take pending' },
        h('div', { class: 'take-head' }, `Review take ${t.take}`,
          h('span', { class: 'take-meta' }, `${fmt(t.duration_s)} · ${fmtBytes(t.size_bytes)}`)),
        h('audio', { controls: true, preload: 'metadata', src: blobUrl(t.blob) }),
        h('div', { class: 'row' },
          h('button', { type: 'button', class: 'btn primary grow', onclick: () => keepTake(t) }, 'Keep'),
          h('button', { type: 'button', class: 'btn grow', onclick: () => retake(t) }, 'Retake'),
          h('button', { type: 'button', class: 'btn', onclick: () => discard(t) }, 'Discard')),
      ));
    }
    const kept = keptFor(sc.id);
    const pk = primaryKey(sc.id);
    for (const t of kept.slice().reverse()) {
      wrap.append(h('div', { class: 'take' },
        h('div', { class: 'take-head' }, `Take ${t.take}`,
          t.key === pk ? h('span', { class: 'badge done' }, 'primary') : null,
          h('span', { class: 'take-meta' }, `${fmt(t.duration_s)} · ${fmtBytes(t.size_bytes)} · ${fmtWhen(t.recorded_at)}`)),
        h('audio', { controls: true, preload: 'none', src: blobUrl(t.blob) }),
        h('div', { class: 'row' },
          h('button', { type: 'button', class: 'btn', onclick: () => exportTake(t) }, 'Share / save'),
          armedButton('Delete', 'Tap again to delete', () => deleteKept(t))),
      ));
    }
    if (kept.length && !pendingFor(sc.id).length) {
      wrap.append(h('p', { class: 'hint' }, 'Done. Record again to add another take (the newest kept take becomes primary), or go to the next scenario.'));
    }
    return wrap;
  }

  // A button that needs two taps within 4 s; replaces window.confirm.
  function armedButton(label, armedLabel, action) {
    let timer = null;
    const b = h('button', { type: 'button', class: 'btn danger' }, label);
    b.addEventListener('click', () => {
      if (b.classList.contains('armed')) { clearTimeout(timer); action(); return; }
      b.classList.add('armed'); b.textContent = armedLabel;
      timer = setTimeout(() => { b.classList.remove('armed'); b.textContent = label; }, 4000);
    });
    return b;
  }

  function renderOverview() {
    const root = $('view-overview');
    const byLen = {};
    for (const s of S.scenarios) {
      const L = s.length || 'other';
      byLen[L] = byLen[L] || [0, 0];
      byLen[L][1]++;
      if (isDone(s.id)) byLen[L][0]++;
    }
    const known = new Set(S.scenarios.map((s) => s.id));
    const orphans = S.takes.filter((t) => !known.has(t.scenario_id)).length;
    const skippedOpen = S.scenarios.filter((s) => S.skipped.has(s.id) && !isDone(s.id)).length;
    const summary = h('div', { class: 'ov-summary' },
      h('span', { class: 'badge done' }, `${doneCount()}/${S.scenarios.length} done`),
      Object.entries(byLen).map(([L, [d, n]]) => h('span', { class: `badge ${L}` }, `${L} ${d}/${n}`)),
      skippedOpen ? h('span', { class: 'badge skipped' }, `${skippedOpen} skipped`) : null,
    );
    const first = S.scenarios.findIndex((s) => !isDone(s.id) && !S.skipped.has(s.id));
    const list = h('ul', { class: 'ov-list' }, S.scenarios.map((s, i) => {
      const done = isDone(s.id);
      const pend = pendingFor(s.id).length > 0;
      const skipped = S.skipped.has(s.id) && !done;
      const dotClass = done ? 'done' : pend ? 'pending' : skipped ? 'skipped' : '';
      const dotText = done ? '✓' : pend ? '!' : skipped ? '↷' : '';
      const n = keptFor(s.id).length;
      const sub = [s.length, s.use_case && String(s.use_case).replace(/_/g, ' '), rangeText(s),
        done ? `${n} take${n > 1 ? 's' : ''}` : pend ? 'needs review' : skipped ? 'skipped' : null].filter(Boolean).join(' · ');
      return h('li', null, h('button', {
        type: 'button', class: `ov-item${i === S.idx ? ' current' : ''}`, onclick: () => go(i),
      },
      h('span', { class: `ov-dot ${dotClass}`, 'aria-hidden': 'true' }, dotText),
      h('span', { class: 'ov-text' },
        h('span', { class: 'ov-title' }, `${s.id} · ${s.title || ''}`),
        h('span', { class: 'ov-sub' }, sub))));
    }));
    setKids(root,
      summary,
      first >= 0 ? h('div', { class: 'row', style: 'margin-bottom:12px' },
        h('button', { type: 'button', class: 'btn primary grow', onclick: () => go(first) }, `Next to record: ${S.scenarios[first].id}`)) : null,
      list,
      orphans ? h('p', { class: 'hint' }, `${orphans} stored take(s) belong to scenario ids not in the current file. They are still exported.`) : null,
    );
  }

  async function renderExport() {
    const root = $('view-export');
    const kept = S.takes.filter((t) => t.status === 'kept');
    const pending = S.takes.filter((t) => t.status === 'pending');
    const bytes = kept.reduce((a, t) => a + (t.size_bytes || 0), 0);
    const covered = new Set(kept.map((t) => t.scenario_id)).size;
    const since = S.lastExportAt ? kept.filter((t) => (t.kept_at || t.recorded_at) > S.lastExportAt).length : kept.length;

    const zipArea = h('div', { class: 'row' });
    if (S.zip) {
      if (S.zip.shareable) {
        zipArea.append(h('button', { type: 'button', class: 'btn primary grow', onclick: shareZip }, 'Share zip'));
      }
      zipArea.append(h('a', {
        class: `btn grow${S.zip.shareable ? '' : ' primary'}`, href: S.zip.url, download: S.zip.name,
        onclick: () => markExported(),
      }, 'Download zip'));
    } else {
      zipArea.append(h('button', { type: 'button', class: 'btn primary grow', id: 'btn-build', disabled: !kept.length, onclick: buildZip }, 'Export zip'));
    }

    const storage = h('p', { class: 'hint', id: 'storage-line' }, 'Storage: checking…');

    const clearPanel = h('section', { class: 'panel' },
      h('h2', null, 'Clear all after export'),
      h('p', null, 'Deletes every take and the skipped/position state from this phone. Scenario cards are not affected.'),
    );
    if (!S.confirmClear) {
      clearPanel.append(h('button', {
        type: 'button', class: 'btn danger', disabled: !S.takes.length,
        onclick: () => { S.confirmClear = true; renderExport(); },
      }, 'Clear all recordings…'));
    } else {
      clearPanel.append(h('div', { class: 'confirm' },
        h('p', null, h('b', null, `Delete all ${S.takes.length} take(s)?`), ' This cannot be undone.'),
        h('p', null, `Last export: ${fmtWhen(S.lastExportAt)}.`),
        since ? h('p', { style: 'color:var(--danger);font-weight:600' }, `${since} kept take(s) are newer than the last export.`) : null,
        pending.length ? h('p', { style: 'color:var(--danger)' }, `${pending.length} take(s) are still awaiting review and were never exported.`) : null,
        h('div', { class: 'row' },
          h('button', { type: 'button', class: 'btn danger armed grow', onclick: clearAll }, 'Delete everything'),
          h('button', { type: 'button', class: 'btn grow', onclick: () => { S.confirmClear = false; renderExport(); } }, 'Cancel')),
      ));
    }

    setKids(root,
      S.isSample ? h('p', { class: 'banner' }, 'Using SAMPLE scenarios.') : null,
      h('section', { class: 'panel' },
        h('h2', null, 'Export'),
        h('div', { class: 'stats' },
          h('div', { class: 'stat' }, h('b', null, String(kept.length)), h('span', null, 'kept takes')),
          h('div', { class: 'stat' }, h('b', null, `${covered}/${S.scenarios.length}`), h('span', null, 'scenarios covered')),
          h('div', { class: 'stat' }, h('b', null, fmtBytes(bytes)), h('span', null, 'audio')),
          h('div', { class: 'stat' }, h('b', null, String(since)), h('span', null, 'new since last export')),
        ),
        h('p', { class: 'hint' }, `Zip contains takes/<id>_t<n>.<ext>, manifest.json and scenarios.json (version ${S.data ? S.data.version : '?'}, from ${S.source}). Takes awaiting review are not included.`),
        pending.length ? h('p', { class: 'hint', style: 'color:var(--under)' }, `${pending.length} take(s) still awaiting Keep/Retake.`) : null,
        zipArea,
        S.zip ? h('p', { class: 'hint' }, `${S.zip.name} · ${fmtBytes(S.zip.blob.size)} · ${S.zip.count} takes. `,
          h('button', { type: 'button', class: 'link-btn', onclick: () => { dropZip(); renderExport(); } }, 'Rebuild')) : null,
        h('p', { class: 'hint' }, `Last export: ${fmtWhen(S.lastExportAt)}`),
        storage,
      ),
      clearPanel,
      h('p', { class: 'hint' }, `App ${APP_VERSION} · single takes can be shared from each scenario card.`),
    );
    fillStorageLine();
  }

  async function fillStorageLine() {
    const el = $('storage-line');
    if (!el) return;
    let txt = 'Storage: ';
    try {
      if (navigator.storage && navigator.storage.persisted) S.persisted = await navigator.storage.persisted();
      txt += S.persisted ? 'persistent (will not be evicted)' : 'best-effort (browser may evict under pressure; export regularly)';
      if (navigator.storage && navigator.storage.estimate) {
        const e = await navigator.storage.estimate();
        txt += ` · using ${fmtBytes(e.usage)} of ${fmtBytes(e.quota)}`;
      }
    } catch (e) { txt += 'unknown'; }
    el.textContent = txt;
  }

  function updateBar() {
    const sc = current();
    const rec = $('btn-rec');
    const timer = $('timer');
    $('btn-prev').disabled = !!S.rec || S.idx <= 0;
    $('btn-next').disabled = !!S.rec || S.idx >= S.scenarios.length - 1;
    if (S.rec) {
      rec.textContent = 'Stop';
      rec.classList.add('recording');
      rec.disabled = false;
      return;
    }
    rec.classList.remove('recording');
    timer.className = 'timer';
    $('meter').style.width = '0';
    if (!sc) { rec.disabled = true; rec.textContent = 'Record'; timer.textContent = ''; return; }
    const pend = pendingFor(sc.id).length > 0;
    const n = keptFor(sc.id).length;
    rec.disabled = pend || S.view !== 'card';
    rec.textContent = pend ? 'Keep or retake first' : n ? `Record take ${n + 1}` : 'Record';
    const rt = rangeText(sc);
    timer.textContent = rt ? `${sc.id} · target ${rt}` : sc.id;
  }

  // ---------- Recording ----------
  function pickMime() {
    if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return '';
    return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) || '';
  }

  async function startRecording() {
    const sc = current();
    if (!sc || S.rec) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      toast('This browser cannot record audio here (needs HTTPS and MediaRecorder).', 6000);
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
    } catch (e) {
      toast(`Microphone unavailable: ${e.name || e.message}`, 6000);
      return;
    }
    const mime = pickMime();
    let recorder;
    try {
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime, audioBitsPerSecond: AUDIO_BITRATE } : { audioBitsPerSecond: AUDIO_BITRATE });
    } catch (e) {
      recorder = new MediaRecorder(stream);
    }
    const track = stream.getAudioTracks()[0];
    let settings = {};
    try {
      const st = track.getSettings ? track.getSettings() : {};
      for (const k of ['echoCancellation', 'noiseSuppression', 'autoGainControl', 'sampleRate', 'sampleSize', 'channelCount', 'latency']) {
        if (st[k] !== undefined) settings[k] = st[k];
      }
    } catch (e) { settings = {}; }

    const r = {
      recorder, stream, chunks: [], scenario_id: sc.id, requestedMime: mime, settings,
      startedAt: new Date().toISOString(), t0: 0, dur: 0, interval: null, ctx: null, raf: 0, wake: null,
    };
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) r.chunks.push(e.data); };
    recorder.onstop = () => finishRecording(r);
    recorder.onerror = (e) => toast(`Recorder error: ${(e.error && e.error.name) || 'unknown'}`, 6000);
    S.rec = r;
    recorder.start(1000);
    r.t0 = performance.now();
    r.interval = setInterval(() => tick(sc), 200);
    tick(sc);
    startMeter(r);
    try { if (navigator.wakeLock) r.wake = await navigator.wakeLock.request('screen'); } catch (e) { /* optional */ }
    if (S.persisted === false) requestPersist();
    updateBar();
  }

  function tick(sc) {
    if (!S.rec) return;
    const sec = (performance.now() - S.rec.t0) / 1000;
    const t = $('timer');
    const rt = rangeText(sc);
    t.textContent = rt ? `● ${fmt(sec)}  /  ${rt}` : `● ${fmt(sec)}`;
    t.className = `timer ${rangeClass(sc, sec)}`;
  }

  function startMeter(r) {
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      r.ctx = new AC();
      const src = r.ctx.createMediaStreamSource(r.stream);
      const an = r.ctx.createAnalyser();
      an.fftSize = 1024;
      src.connect(an);
      const buf = new Float32Array(an.fftSize);
      const fill = $('meter');
      const loop = () => {
        if (S.rec !== r) return;
        an.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        const db = 20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-9);
        fill.style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
        r.raf = requestAnimationFrame(loop);
      };
      loop();
    } catch (e) { /* meter is optional */ }
  }

  function stopRecording() {
    const r = S.rec;
    if (!r) return;
    r.dur = (performance.now() - r.t0) / 1000;
    clearInterval(r.interval);
    try { r.recorder.stop(); } catch (e) { finishRecording(r); }
  }

  async function finishRecording(r) {
    if (r.finished) return;
    r.finished = true;
    cancelAnimationFrame(r.raf);
    r.stream.getTracks().forEach((t) => t.stop());
    if (r.ctx) r.ctx.close().catch(() => {});
    if (r.wake) r.wake.release().catch(() => {});
    const mimeType = r.recorder.mimeType || r.requestedMime || (r.chunks[0] && r.chunks[0].type) || 'audio/webm';
    const blob = new Blob(r.chunks, { type: mimeType });
    S.rec = null;
    if (!blob.size) { toast('Nothing was recorded.'); render(); return; }
    const existing = takesFor(r.scenario_id);
    const take = existing.reduce((m, t) => Math.max(m, t.take), 0) + 1;
    const t = {
      scenario_id: r.scenario_id,
      take,
      status: 'pending',
      mimeType,
      ext: extFor(mimeType),
      duration_s: Math.round(r.dur * 100) / 100,
      size_bytes: blob.size,
      recorded_at: r.startedAt,
      kept_at: null,
      user_agent: navigator.userAgent,
      requested_constraints: AUDIO_CONSTRAINTS,
      audio_settings: r.settings,
      requested_bitrate: AUDIO_BITRATE,
      app_version: APP_VERSION,
      blob,
    };
    try {
      t.key = await dbPutTake(t);
    } catch (e) {
      toast(`Could not save take to IndexedDB: ${e.message}`, 8000);
      t.key = `mem-${Date.now()}`;
    }
    S.takes.push(t);
    render();
  }

  async function keepTake(t) {
    t.status = 'kept';
    t.kept_at = new Date().toISOString();
    try { await dbPutTake(t); } catch (e) { toast(`Save failed: ${e.message}`, 8000); }
    if (S.skipped.delete(t.scenario_id)) dbSetMeta('skipped', [...S.skipped]).catch(() => {});
    dropZip();
    toast(`Kept take ${t.take} as primary.`);
    render();
  }

  async function removeTake(t) {
    try { if (typeof t.key === 'number') await dbDeleteTake(t.key); } catch (e) { toast(`Delete failed: ${e.message}`); return false; }
    S.takes = S.takes.filter((x) => x !== t);
    return true;
  }
  async function discard(t) { if (await removeTake(t)) render(); }
  async function retake(t) { if (await removeTake(t)) { render(); startRecording(); } }
  async function deleteKept(t) { if (await removeTake(t)) { dropZip(); render(); } }

  // ---------- Export ----------
  function buildManifest(kept) {
    const order = new Map(S.scenarios.map((s, i) => [s.id, i]));
    const sorted = kept.slice().sort((a, b) =>
      ((order.has(a.scenario_id) ? order.get(a.scenario_id) : 1e9) - (order.has(b.scenario_id) ? order.get(b.scenario_id) : 1e9))
      || String(a.scenario_id).localeCompare(String(b.scenario_id)) || a.take - b.take);
    const prim = new Set(S.scenarios.map((s) => primaryKey(s.id)));
    // primary for takes whose scenario id is no longer in the file
    const byId = {};
    for (const t of sorted) byId[t.scenario_id] = t.key;
    for (const k of Object.values(byId)) prim.add(k);
    return {
      format: 'dictation-scenarios-export',
      format_version: 1,
      app_version: APP_VERSION,
      exported_at: new Date().toISOString(),
      scenarios_version: S.data ? S.data.version : null,
      scenarios_source: S.source,
      scenarios_file: 'scenarios.json',
      scenario_count: S.scenarios.length,
      scenarios_with_takes: new Set(sorted.map((t) => t.scenario_id)).size,
      scenarios_missing: S.scenarios.filter((s) => !isDone(s.id)).map((s) => s.id),
      takes_count: sorted.length,
      takes: sorted.map((t) => ({
        file: `takes/${takeFileName(t)}`,
        scenario_id: t.scenario_id,
        take: t.take,
        primary: prim.has(t.key),
        mimeType: t.mimeType,
        duration_s: t.duration_s,
        size_bytes: t.size_bytes,
        recorded_at: t.recorded_at,
        kept_at: t.kept_at,
        user_agent: t.user_agent,
        requested_constraints: t.requested_constraints,
        audio_settings: t.audio_settings,
        requested_bitrate: t.requested_bitrate,
      })),
    };
  }

  function dropZip() {
    if (S.zip && S.zip.url) URL.revokeObjectURL(S.zip.url);
    S.zip = null;
  }

  async function buildZip() {
    if (typeof JSZip === 'undefined') { toast('JSZip did not load (offline on first visit?). Reload when online.', 6000); return; }
    const kept = S.takes.filter((t) => t.status === 'kept');
    if (!kept.length) return;
    const btn = $('btn-build');
    if (btn) { btn.disabled = true; btn.textContent = 'Building…'; }
    try {
      const zip = new JSZip();
      const manifest = buildManifest(kept);
      for (const t of kept) zip.file(`takes/${takeFileName(t)}`, t.blob, { compression: 'STORE', date: new Date(t.recorded_at) });
      zip.file('manifest.json', JSON.stringify(manifest, null, 2));
      zip.file('scenarios.json', S.raw);
      const blob = await zip.generateAsync({ type: 'blob', mimeType: 'application/zip', compression: 'DEFLATE' }, (m) => {
        if (btn) btn.textContent = `Building… ${Math.round(m.percent)}%`;
      });
      const stamp = manifest.exported_at.replace(/[-:]/g, '').replace(/\..*$/, '').replace('T', '-');
      const name = `dictation-takes-${stamp}.zip`;
      dropZip();
      const file = new File([blob], name, { type: 'application/zip' });
      let shareable = false;
      try { shareable = !!(navigator.canShare && navigator.canShare({ files: [file] })); } catch (e) { shareable = false; }
      S.zip = { blob, file, name, url: URL.createObjectURL(blob), count: kept.length, shareable };
    } catch (e) {
      toast(`Zip failed: ${e.message}`, 8000);
    }
    renderExport();
  }

  async function shareZip() {
    if (!S.zip) return;
    try {
      await navigator.share({ files: [S.zip.file], title: S.zip.name });
      markExported();
    } catch (e) {
      if (e.name !== 'AbortError') toast(`Share failed (${e.name}). Use Download instead.`, 6000);
    }
  }

  function markExported() {
    S.lastExportAt = new Date().toISOString();
    dbSetMeta('lastExportAt', S.lastExportAt).catch(() => {});
    setTimeout(() => { if (S.view === 'export') renderExport(); }, 300);
  }

  async function exportTake(t) {
    const name = takeFileName(t);
    const file = new File([t.blob], name, { type: (t.mimeType || '').split(';')[0] || 'application/octet-stream' });
    let can = false;
    try { can = !!(navigator.canShare && navigator.canShare({ files: [file] })); } catch (e) { can = false; }
    if (can) {
      try { await navigator.share({ files: [file], title: name }); return; } catch (e) {
        if (e.name === 'AbortError') return;
      }
    }
    const url = URL.createObjectURL(t.blob);
    const a = h('a', { href: url, download: name, style: 'display:none' });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    toast(`Saved ${name} to Downloads.`);
  }

  async function clearAll() {
    try {
      await dbClearTakes();
      await dbClearMeta();
    } catch (e) { toast(`Clear failed: ${e.message}`, 6000); return; }
    S.takes = [];
    S.skipped = new Set();
    S.lastExportAt = null;
    S.confirmClear = false;
    S.idx = 0;
    dropZip();
    toast('All recordings deleted.');
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
