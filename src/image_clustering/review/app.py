"""The single-page review client served by the local review server.

The client is keyboard-first: every shortcut has a visible control carrying the
same key in its tooltip, and the shortcut list stays on screen rather than hiding
behind hover-only affordances.

Box geometry lives in a single authoritative draft in client state. Drag handlers
read that draft rather than a copy captured at render time, so a new drag started
while a previous save is still in flight cannot resurrect stale coordinates.
"""

from __future__ import annotations

# The embedded stylesheet, markup, and client script stay readable as web assets
# rather than being wrapped to Python line limits.
# ruff: noqa: E501

STYLE = """
:root{--bg:#eef1f5;--panel:#fff;--line:#d5dce5;--ink:#17212b;--muted:#5b6b7b;--accent:#1f6feb;
  --page:#2fb344;--occluder:#f03e3e;--unlabelled:#748ffc;--stage:#1b2027}
*{box-sizing:border-box}
body{margin:0;font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
code{font-size:.85em}
kbd{font:600 .7rem/1 ui-monospace,Consolas,monospace;padding:.15rem .3rem;border:1px solid #b8c4d1;border-bottom-width:2px;
  border-radius:.25rem;background:#fff;color:#33414f;margin-left:.15rem}
button{font:inherit;font-size:.85rem;padding:.35rem .6rem;border:1px solid #c3cdd8;border-radius:.4rem;background:#fff;
  color:var(--ink);cursor:pointer;display:inline-flex;align-items:center;gap:.3rem;white-space:nowrap}
button:hover{background:#f1f5fa;border-color:#9fb0c2}
button:focus-visible{outline:3px solid #ffd166;outline-offset:1px}
button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
button[aria-pressed=true] kbd{background:rgba(255,255,255,.2);color:#fff;border-color:rgba(255,255,255,.5)}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.danger{color:#b42318;border-color:#f1b5b0}
button.danger:hover{background:#fdecea}
select,input[type=search]{font:inherit;font-size:.85rem;padding:.3rem .4rem;border:1px solid #c3cdd8;border-radius:.35rem;background:#fff}
/* top bar */
header.top{position:sticky;top:0;z-index:20;display:flex;gap:1rem;align-items:center;padding:.5rem 1rem;
  background:#111a24;color:#e8eef5;box-shadow:0 1px 0 rgba(255,255,255,.06),0 2px 10px rgba(0,0,0,.25)}
header.top h1{font-size:.95rem;font-weight:700;margin:0;letter-spacing:.02em;white-space:nowrap}
details.filters summary,#queue-position{white-space:nowrap}
header.top button{background:#1f2b38;border-color:#33465a;color:#e8eef5}
header.top button:hover{background:#2a3a4b}
header.top kbd{background:#2a3a4b;color:#cfdbe7;border-color:#46607a}
.nav{display:flex;gap:.4rem;align-items:center}
#queue-position{font-weight:600;min-width:9rem;text-align:center;font-variant-numeric:tabular-nums}
details.filters{position:relative}
details.filters summary{cursor:pointer;list-style:none;padding:.35rem .6rem;border:1px solid #33465a;border-radius:.4rem;background:#1f2b38;font-size:.85rem}
details.filters[open] summary{background:#2a3a4b}
.filter-pop{position:absolute;top:2.4rem;left:0;display:grid;grid-template-columns:auto 1fr;gap:.45rem .7rem;align-items:center;
  padding:.8rem;background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:.5rem;box-shadow:0 10px 30px rgba(0,0,0,.25);min-width:22rem}
.filter-pop label{font-size:.78rem;font-weight:600;color:var(--muted)}
.spacer{flex:1}
.save-state{font-size:.8rem;font-weight:600}
.save-state.ok{color:#7ee2a8}
.save-state.pending{color:#ffd166}
.save-state.error{color:#ff9a9a}
/* layout */
main{display:grid;grid-template-columns:minmax(0,1fr) 21rem;gap:1rem;padding:1rem;align-items:start}
.workspace{min-width:0}
.progress{font-size:.78rem;color:var(--muted);margin:0 0 .6rem}
.cluster-head{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:0 0 .25rem}
.cluster-head h2{font-size:1.05rem;margin:0 .3rem 0 0}
.meta{margin:0 0 .7rem;font-size:.8rem;color:var(--muted)}
.chip{display:inline-flex;align-items:center;gap:.25rem;padding:.08rem .5rem;border-radius:1rem;font-size:.72rem;font-weight:600;
  border:1px solid #c3cdd8;background:#f4f6f9;color:#33414f;white-space:nowrap}
.chip.approved,.chip.same_document{background:#e3f7e8;border-color:#7cc791;color:#1d6b33}
.chip.edited{background:#fff4d6;border-color:#e2b74b;color:#7a5a00}
.chip.dissolved,.chip.different_document{background:#fde8e6;border-color:#eb9a92;color:#a1281c}
.chip.irregular,.chip.unsure{background:#efe8ff;border-color:#a991f0;color:#5636b8}
.chip.same_background_document{background:#e3efff;border-color:#86b0f2;color:#1b4fa8}
.chip.unset{border-style:dashed;background:#fff;color:var(--muted)}
.chip.star{background:#fff3c4;border-color:#d4a106;color:#6b5000}
.members{display:grid;grid-template-columns:repeat(auto-fit,minmax(20rem,1fr));gap:.8rem}
.members.single{grid-template-columns:minmax(0,1fr)}
.member{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;overflow:hidden;box-shadow:0 1px 2px rgba(16,24,40,.05)}
.member.focused{border-color:var(--accent);box-shadow:0 0 0 2px rgba(31,111,235,.35)}
.member.excluded .stage img{opacity:.35;filter:grayscale(1)}
.member.primary{border-color:#d4a106;box-shadow:0 0 0 2px rgba(212,161,6,.45)}
.member-head{display:flex;gap:.5rem;align-items:center;padding:.45rem .6rem;font-size:.8rem;border-bottom:1px solid var(--line)}
.member-head .num{font-weight:700}
.member-head .name{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;flex:1}
.stage{position:relative;display:flex;justify-content:center;background:var(--stage);padding:1.4rem .6rem .6rem;
  -webkit-user-select:none;user-select:none;touch-action:none}
.image-canvas{position:relative;width:fit-content;max-width:100%;line-height:0}
.stage img{display:block;max-width:100%;width:auto;height:auto;max-height:calc(100vh - 13rem)}
.single .stage img{max-height:calc(100vh - 12rem)}
.stage.editing,.stage.editing .shape{cursor:crosshair}
.member-foot{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;padding:.45rem .6rem;font-size:.78rem}
.member-foot a{color:var(--muted);margin-left:auto}
.member-foot button{font-size:.78rem;padding:.25rem .5rem}
/* overlays: one SVG of shapes per capture, HTML handles and tags on top */
.overlay{position:absolute;left:0;top:0;width:100%;height:100%;overflow:visible;pointer-events:none}
.overlay .shape{pointer-events:all;cursor:move;stroke-width:2;vector-effect:non-scaling-stroke;fill-opacity:.1;stroke-linejoin:round}
.overlay .shape.selected{stroke-width:3.5;fill-opacity:.2}
.overlay .bounds{fill:none;stroke-width:1.2;stroke-dasharray:6 5;vector-effect:non-scaling-stroke;opacity:.9}
.overlay .kind-base_page{stroke:var(--page);fill:var(--page)}
.overlay .kind-data_bearing_overlay{stroke:var(--occluder);fill:var(--occluder)}
.overlay .kind-reviewer{stroke:var(--unlabelled);fill:var(--unlabelled)}
.overlay .bounds{fill:none}
.kind-base_page{--c:var(--page)}
.kind-data_bearing_overlay{--c:var(--occluder)}
.kind-reviewer{--c:var(--unlabelled)}
.tag{position:absolute;transform:translateY(-100%);font:600 11px/1.55 system-ui,sans-serif;padding:0 .4rem;border-radius:.25rem .25rem .25rem 0;
  color:#fff;background:var(--c);white-space:nowrap;pointer-events:none;z-index:2;box-shadow:0 1px 3px rgba(0,0,0,.4)}
.tag.inside{transform:none;border-radius:0 0 .25rem 0}
.tag .src{font-weight:400;opacity:.9}
.handle{position:absolute;z-index:3;width:10px;height:10px;transform:translate(-50%,-50%);border-radius:50%;background:#fff;
  border:2px solid var(--c);box-shadow:0 0 0 1px rgba(0,0,0,.45);cursor:grab}
.handle:hover,.handle.active{width:16px;height:16px}
.handle.selected{width:14px;height:14px}
.handle.side{border-radius:2px;width:10px;height:10px;background:var(--c);border-color:#fff}
.handle.side.selected{width:11px;height:11px}
/* side panel */
aside{position:sticky;top:3.6rem;display:flex;flex-direction:column;gap:.75rem;max-height:calc(100vh - 4.6rem);overflow:auto;padding-bottom:.5rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;padding:.75rem .8rem;box-shadow:0 1px 2px rgba(16,24,40,.05)}
.card h3{font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 .55rem}
.card p.hint{font-size:.75rem;color:var(--muted);margin:.5rem 0 0}
.relations{display:grid;grid-template-columns:1fr 1fr;gap:.4rem}
.relations button{flex-direction:column;align-items:flex-start;gap:.1rem;padding:.45rem .55rem;white-space:normal;text-align:left;border-left-width:4px}
.relations button small{font-size:.7rem;color:var(--muted);font-weight:400}
.relations button b{display:flex;justify-content:space-between;width:100%;font-weight:600}
#rel-same{border-left-color:#3fa96c}
#rel-background{border-left-color:#1f6feb}
#rel-different{border-left-color:#d64535}
#rel-unsure{border-left-color:#7950f2}
.relations button[aria-pressed=true]{background:#eef4ff;color:var(--ink);border-color:#9fb0c2;box-shadow:inset 0 0 0 1px var(--accent)}
.relations button[aria-pressed=true] small{color:var(--muted)}
.row{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.5rem}
.row button{font-size:.8rem;padding:.3rem .5rem}
.box-list{display:flex;flex-direction:column;gap:.3rem}
.box-row{display:grid;grid-template-columns:auto 1fr auto;gap:.5rem;align-items:center;padding:.35rem .5rem;border:1px solid var(--line);
  border-radius:.4rem;cursor:pointer;font-size:.8rem;background:#fff;text-align:left;white-space:normal;width:100%}
.box-row.selected{border-color:var(--accent);background:#eef4ff}
.box-row .sw{width:14px;height:14px;border-radius:3px;border:2px solid var(--c);background:color-mix(in srgb,var(--c) 20%,transparent)}
.box-row .what{font-weight:600}
.box-row .how{display:block;font-size:.72rem;color:var(--muted);font-weight:400}
.box-row .conf{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.75rem}
.empty-boxes{font-size:.8rem;color:var(--muted)}
.legend{display:grid;grid-template-columns:44px 1fr;gap:.45rem .6rem;align-items:center;font-size:.78rem}
.legend svg{display:block}
.legend .handle-demo{position:relative;width:44px;height:14px}
.legend .handle-demo .handle{position:absolute;top:7px}
details.card summary{cursor:pointer;font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
details.card table{width:100%;border-collapse:collapse;font-size:.76rem;margin-top:.5rem}
details.card td{padding:.15rem 0;vertical-align:top}
details.card td:first-child{white-space:nowrap;padding-right:.5rem}
/* The unsaved-edit bar floats over the page, so showing or hiding it never moves
   the progress line or image cards and it takes no space while hidden. */
.draft-banner{position:fixed;left:50%;bottom:1rem;transform:translateX(-50%);z-index:30;display:flex;gap:.5rem;align-items:center;
  padding:.45rem .7rem;background:#fff4d1;border:1px solid #e2b74b;border-radius:.5rem;font-size:.82rem;font-weight:600;
  box-shadow:0 8px 24px rgba(0,0,0,.25);white-space:nowrap}
.draft-banner[hidden]{display:none}
.optional-toggle{width:100%;margin-top:.4rem;flex-direction:column;align-items:flex-start;gap:.1rem;white-space:normal;text-align:left;border-style:dashed}
.optional-toggle b{display:flex;justify-content:space-between;width:100%;font-weight:600}
.optional-toggle small{font-size:.7rem;color:var(--muted)}
.optional-toggle[aria-pressed=true]{background:#f3f0ff;color:var(--ink);border:1px solid #7950f2;box-shadow:inset 0 0 0 1px #7950f2}
.chip.optional{background:#f3f0ff;border:1px dashed #7950f2;color:#5636b8}
.overlay .shape.optional{stroke-dasharray:1.5 4;stroke-linecap:round;fill-opacity:.04}
.legend-strip{display:flex;flex-wrap:wrap;gap:.3rem 1rem;align-items:center;margin:0 0 .6rem;font-size:.75rem;color:var(--muted)}
.legend-strip span{display:inline-flex;align-items:center;gap:.3rem}
.legend-strip svg{display:block}
.legend-strip button{font-size:.72rem;padding:.1rem .45rem}
.filmstrip{display:flex;gap:.4rem;overflow-x:auto;padding:.6rem 0 0}
.filmstrip button{padding:.15rem;border-width:2px;flex:0 0 auto}
.filmstrip button[aria-current=true]{border-color:var(--accent)}
.filmstrip img{display:block;height:4.5rem;width:auto}
.filmstrip .excluded-thumb{opacity:.4}
.empty{padding:2rem;text-align:center;color:var(--muted)}
/* zoom loupe shown while a corner, edge or new box is dragged */
#loupe{position:fixed;z-index:50;width:220px;height:220px;border-radius:50%;overflow:hidden;pointer-events:none;
  border:3px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.5),0 10px 30px rgba(0,0,0,.45);background:#111}
#loupe[hidden]{display:none}
#loupe canvas{display:block;width:220px;height:220px}
#loupe-label{position:absolute;left:0;right:0;bottom:14px;text-align:center;font:600 10px/1 ui-monospace,Consolas,monospace;color:#fff;
  text-shadow:0 0 3px #000,0 0 3px #000}
@media (max-width:1000px){main{grid-template-columns:minmax(0,1fr)}aside{position:static;max-height:none}}
"""

SCRIPT = r"""
const state = {
  dataset: null,
  decisions: {},
  progress: null,
  queue: [],
  index: 0,
  focus: 0,
  viewMode: 'grid',
  editing: false,
  selectedBox: null,
  draft: null,
  dragging: false,
  undoStack: [],
  commitTimer: null,
};

const el = (id) => document.getElementById(id);
const saveState = el('save-state');

function setSave(kind, message) {
  saveState.className = `save-state ${kind}`;
  saveState.textContent = message;
}

async function api(path, options) {
  setSave('pending', 'Saving...');
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    setSave('error', payload.error || 'Save failed');
    throw new Error(payload.error || 'request failed');
  }
  if (payload.progress) state.progress = payload.progress;
  if (payload.cluster) state.decisions[payload.cluster.cluster_id] = payload.cluster;
  setSave('ok', `Saved ${new Date().toLocaleTimeString()}`);
  return payload;
}

function clusterById(clusterId) {
  return state.dataset.clusters.find((cluster) => cluster.cluster_id === clusterId);
}

function emptyDecision() {
  return {status: 'unreviewed', dissolved: false, excluded_image_ids: [], images: {}, relation: null, primary_view: null};
}

function decisionFor(clusterId) {
  return state.decisions[clusterId] || emptyDecision();
}

function imageDecision(clusterId, imageId) {
  return decisionFor(clusterId).images?.[imageId] || {included: true, bbox_status: 'unreviewed', boxes: null};
}

function isExcluded(clusterId, imageId) {
  return decisionFor(clusterId).excluded_image_ids?.includes(imageId) || false;
}

function savedBoxes(cluster, image) {
  const decision = imageDecision(cluster.cluster_id, image.image_id);
  const boxes = decision.boxes === null || decision.boxes === undefined ? image.boxes : decision.boxes;
  return boxes.map((box) => ({...box, bbox: [...box.bbox], polygon: box.polygon ? box.polygon.map((point) => [...point]) : null}));
}

// Single source of truth for geometry: the draft when one is open for this
// image, otherwise the persisted boxes. Never captured in a closure.
function currentBoxes(cluster, image) {
  if (state.draft && state.draft.clusterId === cluster.cluster_id && state.draft.imageId === image.image_id) {
    return state.draft.boxes;
  }
  return savedBoxes(cluster, image);
}

function hasDraft() {
  return Boolean(state.draft);
}

function openDraft(cluster, image) {
  if (state.draft && state.draft.clusterId === cluster.cluster_id && state.draft.imageId === image.image_id) return state.draft;
  state.draft = {clusterId: cluster.cluster_id, imageId: image.image_id, boxes: savedBoxes(cluster, image)};
  return state.draft;
}

function scheduleCommit(delay) {
  window.clearTimeout(state.commitTimer);
  state.commitTimer = window.setTimeout(() => commitDraft(), delay);
}

async function commitDraft() {
  window.clearTimeout(state.commitTimer);
  const draft = state.draft;
  if (!draft) return;
  const cluster = clusterById(draft.clusterId);
  const image = cluster?.images.find((item) => item.image_id === draft.imageId);
  if (!cluster || !image) { state.draft = null; render(); return; }
  pushUndo(cluster.cluster_id);
  const boxes = draft.boxes.map((box) => ({
    ...box,
    bbox: box.bbox.map((value) => Math.round(value)),
    polygon: box.polygon ? box.polygon.map((point) => point.map((value) => Math.round(value))) : null,
  }));
  state.draft = null;
  try {
    const payload = await api(`/api/clusters/${encodeURIComponent(cluster.cluster_id)}/images/${encodeURIComponent(image.image_id)}/boxes`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({boxes}),
    });
    applyImage(cluster.cluster_id, image.image_id, payload.image);
    setSave('ok', `Box edit saved ${new Date().toLocaleTimeString()}`);
  } catch (error) {
    state.undoStack.pop();
    setSave('error', `Box edit rejected: ${error.message}`);
  }
  render();
}

function cancelDraft() {
  window.clearTimeout(state.commitTimer);
  if (!state.draft) return false;
  state.draft = null;
  state.selectedBox = null;
  setSave('ok', 'Box edit cancelled; nothing was saved');
  render();
  return true;
}

function pushUndo(clusterId) {
  const snapshot = JSON.parse(JSON.stringify(state.decisions[clusterId] || emptyDecision()));
  state.undoStack.push({clusterId, record: snapshot});
  if (state.undoStack.length > 50) state.undoStack.shift();
}

async function undo() {
  if (state.draft) { cancelDraft(); return; }
  const entry = state.undoStack.pop();
  if (!entry) { setSave('ok', 'Nothing left to undo'); return; }
  await api(`/api/clusters/${encodeURIComponent(entry.clusterId)}/restore`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cluster: entry.record}),
  });
  const queueIndex = state.queue.indexOf(entry.clusterId);
  if (queueIndex >= 0) state.index = queueIndex;
  setSave('ok', `Undid the last change to ${entry.clusterId}`);
  render();
}

function applyImage(clusterId, imageId, record) {
  const decision = state.decisions[clusterId] || emptyDecision();
  decision.images = decision.images || {};
  decision.images[imageId] = record;
  state.decisions[clusterId] = decision;
}

function rebuildQueue() {
  const query = el('search').value.trim().toLowerCase();
  const minSize = Number(el('min-size').value);
  const sortKey = el('sort').value;
  const statusFilter = el('status-filter').value;
  let clusters = state.dataset.clusters.filter((cluster) => {
    const decision = decisionFor(cluster.cluster_id);
    const haystack = `${cluster.cluster_id} ${cluster.source_folder} ${cluster.review_reasons.join(' ')}`.toLowerCase();
    const confirmed = decision.status === 'approved' || decision.status === 'edited';
    const statusMatch = !statusFilter
      || (statusFilter === 'unreviewed' ? decision.status === 'unreviewed'
        : statusFilter === 'reviewed' ? decision.status !== 'unreviewed'
        : statusFilter === 'needs-relation' ? confirmed && !decision.relation
        : statusFilter === 'needs-primary' ? confirmed && !decision.primary_view
        : decision.status === statusFilter);
    return cluster.image_count >= minSize && (!query || haystack.includes(query)) && statusMatch;
  });
  const confidence = (cluster) => (cluster.minimum_confidence === null ? 2 : cluster.minimum_confidence);
  clusters = clusters.sort((a, b) => {
    if (sortKey === 'confidence-asc') return confidence(a) - confidence(b) || a.cluster_id.localeCompare(b.cluster_id);
    if (sortKey === 'confidence-desc') return confidence(b) - confidence(a) || a.cluster_id.localeCompare(b.cluster_id);
    if (sortKey === 'size-desc') return b.image_count - a.image_count || a.cluster_id.localeCompare(b.cluster_id);
    if (sortKey === 'size-asc') return a.image_count - b.image_count || a.cluster_id.localeCompare(b.cluster_id);
    if (sortKey === 'folder') return a.source_folder.localeCompare(b.source_folder) || a.cluster_id.localeCompare(b.cluster_id);
    return a.cluster_id.localeCompare(b.cluster_id);
  });
  state.queue = clusters.map((cluster) => cluster.cluster_id);
  state.index = Math.min(state.index, Math.max(state.queue.length - 1, 0));
}

function currentCluster() {
  const clusterId = state.queue[state.index];
  return clusterId ? clusterById(clusterId) : null;
}

function focusedImage() {
  const cluster = currentCluster();
  return cluster ? cluster.images[state.focus] : null;
}

function previewUrl(image, size) {
  return `/preview?size=${size}&path=${encodeURIComponent(image.source_path)}`;
}

function renderProgress() {
  const p = state.progress;
  if (p) {
    el('progress').textContent =
      `${p.reviewed_cluster_count} of ${p.cluster_count} clusters reviewed `
      + `(${p.approved_cluster_count} approved, ${p.edited_cluster_count} edited, `
      + `${p.dissolved_cluster_count} dissolved, ${p.irregular_cluster_count} irregular) · `
      + `${p.bbox_approved_image_count} of ${p.image_count} captures with approved boxes`
      + (p.relation_counts
        ? ` · relation: ${p.relation_counts.same_document} same doc, ${p.relation_counts.same_background_document} same background, `
          + `${p.relation_counts.different_document} different, ${p.relation_counts.unsure} unsure · `
          + `${p.primary_view_set_count} with primary view set`
        : '')
      + (p.optional_cluster_count ? ` · ${p.optional_cluster_count} optional` : '');
  }
  el('queue-position').textContent = state.queue.length
    ? `Cluster ${state.index + 1} / ${state.queue.length}`
    : 'No clusters match';
}

// Box kinds shared with the cropper: the page of interest and the covering (foreground) sheet.
const BOX_KIND_LABELS = {base_page: 'page', data_bearing_overlay: 'occluder', reviewer: 'unlabelled'};
const BOX_KIND_NAMES = {base_page: 'Page of interest', data_bearing_overlay: 'Occluder (sheet on top)', reviewer: 'Unlabelled'};
const KIND_COLORS = {base_page: '#2fb344', data_bearing_overlay: '#f03e3e', reviewer: '#748ffc'};
const kindOf = (box) => (KIND_COLORS[box.kind] ? box.kind : 'reviewer');

// A drawn box is the page until the capture has one, then an occluder; t flips the selected box.
function newBoxKind(boxes) {
  return boxes.some((box) => box.kind === 'base_page') ? 'data_bearing_overlay' : 'base_page';
}

function toggleSelectedBoxKind() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (state.selectedBox === null || !cluster || !image) return;
  const draft = openDraft(cluster, image);
  const box = draft.boxes[state.selectedBox];
  if (!box) return;
  box.kind = box.kind === 'base_page' ? 'data_bearing_overlay' : 'base_page';
  commitDraft();
}

// Geometry. A box's shape is its polygon when it has one (a 4-corner quad the reviewer
// edits, or a longer segmentation outline), otherwise its bbox rectangle. The bbox is
// always the shape's bounds. Quads are kept in top-left, top-right, bottom-right,
// bottom-left order so side handles know which corners they move.
const rectQuad = ([x0, y0, x1, y1]) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function shapeOf(box) {
  return box.polygon && box.polygon.length >= 3 ? box.polygon : rectQuad(box.bbox);
}

function boundsOf(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function isRect(points) {
  if (points.length !== 4) return false;
  const [x0, y0, x1, y1] = boundsOf(points);
  return points.every(([x, y]) => (Math.abs(x - x0) < 1 || Math.abs(x - x1) < 1) && (Math.abs(y - y0) < 1 || Math.abs(y - y1) < 1));
}

// The four extreme points of any outline, as an ordered quad; a SAM outline becomes its corners.
function cornersOf(points) {
  const pick = (score) => points.reduce((best, point) => (score(point) < score(best) ? point : best));
  const quad = [pick(([x, y]) => x + y), pick(([x, y]) => y - x), pick(([x, y]) => -(x + y)), pick(([x, y]) => x - y)];
  return new Set(quad).size === 4 ? quad.map((point) => [...point]) : rectQuad(boundsOf(points));
}

function setShape(box, points, image) {
  box.polygon = points.map(([x, y]) => [clamp(x, 0, image.width), clamp(y, 0, image.height)]);
  box.bbox = boundsOf(box.polygon);
}

function shapeLabel(box) {
  const points = shapeOf(box);
  if (isRect(points)) return 'rectangle';
  return points.length === 4 ? '4-corner shape' : `outline, ${points.length} points`;
}

function sourceLabel(box) {
  if (box.origin === 'reviewer') return 'your edit';
  const source = String(box.completeness || '');
  if (source.startsWith('sam:')) return `SAM via ${source.slice(4)}`;
  return source || 'machine box';
}

function confidenceLabel(box) {
  return box.origin !== 'reviewer' && typeof box.confidence === 'number' ? box.confidence.toFixed(2) : '';
}

function area(box) {
  const [x0, y0, x1, y1] = box.bbox;
  return (x1 - x0) * (y1 - y0);
}

const SVG_NS = 'http://www.w3.org/2000/svg';

// One SVG of shapes (largest first, so a sheet on a page stays clickable) plus HTML
// tags and handles. Every box shows its round corner handles, so one drag edits it;
// the selected box also gets square side handles.
function renderBoxes(canvas, image, boxes, editable) {
  for (const node of [...canvas.querySelectorAll('.overlay, .tag, .handle')]) node.remove();
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'overlay');
  svg.setAttribute('viewBox', `0 0 ${image.width} ${image.height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  const order = boxes.map((box, index) => index).sort((a, b) => area(boxes[b]) - area(boxes[a]));
  const pct = ([x, y]) => [`${(x / image.width) * 100}%`, `${(y / image.height) * 100}%`];
  for (const boxIndex of order) {
    const box = boxes[boxIndex];
    const points = shapeOf(box);
    const kind = kindOf(box);
    const selected = editable && state.selectedBox === boxIndex;
    if (!isRect(points)) {
      const [x0, y0, x1, y1] = box.bbox;
      const bounds = document.createElementNS(SVG_NS, 'rect');
      bounds.setAttribute('class', `bounds kind-${kind}`);
      bounds.setAttribute('x', x0); bounds.setAttribute('y', y0);
      bounds.setAttribute('width', x1 - x0); bounds.setAttribute('height', y1 - y0);
      svg.appendChild(bounds);
    }
    const shape = document.createElementNS(SVG_NS, 'polygon');
    shape.setAttribute('class', `shape kind-${kind}${selected ? ' selected' : ''}${box.optional ? ' optional' : ''}`);
    shape.setAttribute('points', points.map((point) => point.join(',')).join(' '));
    shape.dataset.boxIndex = String(boxIndex);
    svg.appendChild(shape);
  }
  canvas.appendChild(svg);
  boxes.forEach((box, boxIndex) => {
    const kind = kindOf(box);
    const selected = editable && state.selectedBox === boxIndex;
    const [x0, y0] = box.bbox;
    const tag = document.createElement('span');
    tag.className = `tag kind-${kind}` + (y0 / image.height < 0.04 ? ' inside' : '');
    [tag.style.left, tag.style.top] = pct([x0, y0]);
    const confidence = confidenceLabel(box);
    tag.innerHTML = `${boxIndex + 1} ${BOX_KIND_LABELS[kind]}${box.optional ? ' (optional)' : ''} <span class='src'>· ${sourceLabel(box)}${confidence ? ` ${confidence}` : ''}</span>`;
    canvas.appendChild(tag);
    const corners = cornersOf(shapeOf(box));
    corners.forEach((point, corner) => {
      const handle = document.createElement('span');
      handle.className = `handle corner kind-${kind}${selected ? ' selected' : ''}`;
      handle.dataset.corner = String(corner);
      handle.dataset.boxIndex = String(boxIndex);
      handle.title = 'Drag this corner (Shift keeps a rectangle)';
      [handle.style.left, handle.style.top] = pct(point);
      canvas.appendChild(handle);
    });
    if (!selected) return;
    corners.forEach((point, side) => {
      const next = corners[(side + 1) % 4];
      const handle = document.createElement('span');
      handle.className = `handle side kind-${kind} selected`;
      handle.dataset.side = String(side);
      handle.dataset.boxIndex = String(boxIndex);
      handle.title = 'Drag to move this side';
      [handle.style.left, handle.style.top] = pct([(point[0] + next[0]) / 2, (point[1] + next[1]) / 2]);
      canvas.appendChild(handle);
    });
  });
}

function renderBoxPanel(cluster) {
  const list = el('box-list');
  list.innerHTML = '';
  const image = cluster ? cluster.images[state.focus] : null;
  el('box-panel-title').textContent = image ? `3 · Boxes on capture ${state.focus + 1}` : '3 · Boxes';
  if (!image) return;
  const boxes = currentBoxes(cluster, image);
  const status = imageDecision(cluster.cluster_id, image.image_id).bbox_status;
  if (!boxes.length) {
    list.innerHTML = "<div class='empty-boxes'>No boxes. Press <kbd>e</kbd> and drag on the image to draw one.</div>";
  }
  boxes.forEach((box, boxIndex) => {
    const kind = kindOf(box);
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `box-row kind-${kind}` + (state.selectedBox === boxIndex ? ' selected' : '');
    row.innerHTML = `<span class='sw'></span><span><span class='what'>${boxIndex + 1} · ${BOX_KIND_NAMES[kind]}${box.optional ? ' · optional' : ''}</span>`
      + `<span class='how'>${shapeLabel(box)} · ${sourceLabel(box)}</span></span><span class='conf'>${confidenceLabel(box)}</span>`;
    row.addEventListener('click', () => { state.selectedBox = boxIndex; render(); });
    list.appendChild(row);
  });
  const approve = el('approve-boxes');
  approve.innerHTML = status === 'approved' ? '✓ Boxes approved <kbd>b</kbd>' : 'Approve boxes <kbd>b</kbd>';
  approve.className = status === 'approved' ? '' : 'primary';
  const none = state.selectedBox === null || state.selectedBox >= boxes.length;
  for (const id of ['box-kind', 'square-box', 'delete-box', 'box-optional']) el(id).disabled = none;
  el('box-optional').setAttribute('aria-pressed', String(!none && Boolean(boxes[state.selectedBox].optional)));
}

function renderDraftBanner() {
  const banner = el('draft-banner');
  banner.hidden = !hasDraft();
  if (hasDraft()) {
    el('draft-summary').textContent =
      `Unsaved box edit on ${state.draft.imageId.split('/').pop()} (${state.draft.boxes.length} boxes)`;
  }
}

const BBOX_STATUS_LABELS = {approved: '✓ boxes approved', edited: 'boxes edited', unreviewed: 'boxes not reviewed'};

function renderMemberCard(cluster, image, imageIndex) {
  const excluded = isExcluded(cluster.cluster_id, image.image_id);
  const imageState = imageDecision(cluster.cluster_id, image.image_id);
  const boxes = currentBoxes(cluster, image);
  const isFocused = imageIndex === state.focus;
  const editable = state.editing && isFocused;
  const isPrimary = decisionFor(cluster.cluster_id).primary_view === image.image_id;
  const card = document.createElement('article');
  card.className = 'member' + (isFocused ? ' focused' : '') + (excluded ? ' excluded' : '') + (isPrimary ? ' primary' : '');
  card.innerHTML =
    `<div class='member-head'><span class='num'>Capture ${imageIndex + 1}<kbd>${imageIndex + 1}</kbd></span>`
    + `<span class='name' title='${image.filename}'>#${image.sequence_index} · ${image.filename}</span>`
    + (isPrimary ? "<span class='chip star'>★ primary unoccluded</span>" : '')
    + `<span class='chip ${imageState.bbox_status}'>${BBOX_STATUS_LABELS[imageState.bbox_status] || imageState.bbox_status}</span>`
    + (excluded ? "<span class='chip dissolved'>excluded</span>" : '')
    + '</div>'
    + `<div class='stage${editable ? ' editing' : ''}'><div class='image-canvas'></div></div>`
    + "<div class='member-foot'></div>";
  const stage = card.querySelector('.stage');
  const canvas = stage.querySelector('.image-canvas');
  const picture = document.createElement('img');
  picture.loading = 'lazy';
  picture.alt = `Capture ${image.filename}`;
  // Use the same edit-sized source for every main capture. Focus changes must
  // not make one side-by-side image sharper than its sibling.
  picture.src = previewUrl(image, 'edit');
  picture.draggable = false;
  canvas.appendChild(picture);
  if (image.width && image.height) renderBoxes(canvas, image, boxes, isFocused);
  const foot = card.querySelector('.member-foot');
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.setAttribute('aria-pressed', String(!excluded));
  toggle.title = `Include this capture in the cluster (key ${imageIndex + 1}, or x when focused)`;
  toggle.textContent = excluded ? 'Excluded' : 'In cluster';
  toggle.addEventListener('click', () => setMembership(cluster, image, excluded));
  foot.appendChild(toggle);
  const primary = document.createElement('button');
  primary.type = 'button';
  primary.setAttribute('aria-pressed', String(isPrimary));
  primary.title = 'Mark this capture as the primary unoccluded view of the document (key p when focused)';
  primary.textContent = isPrimary ? '★ Primary unoccluded' : '☆ Primary view';
  primary.addEventListener('click', () => setPrimaryView(cluster, isPrimary ? null : image.image_id));
  foot.appendChild(primary);
  const approve = document.createElement('button');
  approve.type = 'button';
  approve.title = 'Approve the boxes on this capture (key b)';
  approve.textContent = imageState.bbox_status === 'approved' ? 'Unapprove boxes' : 'Approve boxes';
  approve.addEventListener('click', () => approveBoxes(cluster, image, imageState.bbox_status !== 'approved'));
  foot.appendChild(approve);
  const link = document.createElement('a');
  link.href = previewUrl(image, 'edit');
  link.target = '_blank';
  link.rel = 'noreferrer';
  link.textContent = 'open full preview ↗';
  foot.appendChild(link);
  card.addEventListener('pointerdown', () => {
    if (state.focus !== imageIndex && !state.dragging) {
      if (hasDraft()) commitDraft();
      state.focus = imageIndex;
      state.selectedBox = null;
      render();
    }
  });
  if (image.width && image.height) attachBoxEditing(stage, cluster, image, editable);
  return card;
}

function renderFilmstrip(cluster) {
  const strip = document.createElement('div');
  strip.className = 'filmstrip';
  cluster.images.forEach((image, imageIndex) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-current', String(imageIndex === state.focus));
    button.title = `Show capture ${imageIndex + 1} (${image.filename})`;
    const thumb = document.createElement('img');
    thumb.loading = 'lazy';
    thumb.src = previewUrl(image, 'thumbnail');
    thumb.alt = image.filename;
    if (isExcluded(cluster.cluster_id, image.image_id)) thumb.className = 'excluded-thumb';
    button.appendChild(thumb);
    button.addEventListener('click', () => {
      if (hasDraft()) commitDraft();
      state.focus = imageIndex;
      state.selectedBox = null;
      render();
    });
    strip.appendChild(button);
  });
  return strip;
}

// One line above the captures saying what every mark means; the panel has the long version.
const swatch = (body) => `<svg width='26' height='14'>${body}</svg>`;
const LEGEND_STRIP = "<div class='legend-strip'>"
  + `<span>${swatch("<rect x='1' y='1' width='24' height='12' fill='#2fb344' fill-opacity='.18' stroke='#2fb344' stroke-width='2'/>")}page of interest</span>`
  + `<span>${swatch("<rect x='1' y='1' width='24' height='12' fill='#f03e3e' fill-opacity='.18' stroke='#f03e3e' stroke-width='2'/>")}occluder (sheet on top)</span>`
  + `<span>${swatch("<rect x='1' y='1' width='24' height='12' fill='#748ffc' fill-opacity='.18' stroke='#748ffc' stroke-width='2'/>")}unlabelled</span>`
  + `<span>${swatch("<polygon points='4,1 24,1 25,13 1,13' fill='none' stroke='#5b6b7b' stroke-width='2'/>")}solid = the shape</span>`
  + `<span>${swatch("<rect x='1' y='1' width='24' height='12' fill='none' stroke='#5b6b7b' stroke-width='1.2' stroke-dasharray='4 3'/>")}dashed = its upright bbox</span>`
  + `<span>${swatch("<rect x='1' y='1' width='24' height='12' fill='none' stroke='#5b6b7b' stroke-width='2' stroke-dasharray='1.5 3' stroke-linecap='round'/>")}dotted = optional</span>`
  + `<span>${swatch("<rect x='2' y='2' width='22' height='10' fill='none' stroke='#5b6b7b' stroke-width='3.5'/>")}thick = selected</span>`
  + "<button type='button' onclick=\"const d=document.getElementById('legend-card');d.open=true;d.scrollIntoView({block:'nearest'})\">more…</button>"
  + '</div>';

function render() {
  rebuildQueue();
  renderProgress();
  renderDraftBanner();
  el('view-mode').setAttribute('aria-pressed', String(state.viewMode === 'focus'));
  el('view-mode-label').textContent = state.viewMode === 'focus' ? 'Full screen' : 'Side by side';
  el('main').className = state.viewMode === 'focus' ? 'focus-mode' : '';
  el('edit-mode').setAttribute('aria-pressed', String(state.editing));
  const host = el('cluster-host');
  const cluster = currentCluster();
  if (!cluster) {
    host.innerHTML = "<div class='empty'>No clusters match the current filters.</div>";
    renderBoxPanel(null);
    return;
  }
  state.focus = Math.max(0, Math.min(state.focus, cluster.images.length - 1));
  const decision = decisionFor(cluster.cluster_id);
  for (const [id, relation] of [['rel-same', 'same_document'], ['rel-background', 'same_background_document'],
    ['rel-different', 'different_document'], ['rel-unsure', 'unsure']]) {
    el(id).setAttribute('aria-pressed', String(decision.relation === relation));
  }
  el('optional').setAttribute('aria-pressed', String(Boolean(decision.optional)));
  const confidence = cluster.minimum_confidence === null
    ? 'no accepted registration'
    : `weakest link ${cluster.minimum_confidence.toFixed(3)}`;
  host.innerHTML = '';
  const panel = document.createElement('section');
  panel.className = 'cluster';
  panel.innerHTML =
    `<div class='cluster-head'><h2>${cluster.cluster_id}</h2><span class='chip ${decision.status}'>${decision.status}</span>`
    + `<span class='chip ${decision.relation || 'unset'}'>${RELATION_LABELS[decision.relation] || 'relation not set'}</span>`
    + `<span class='chip ${decision.primary_view ? 'star' : 'unset'}'>${primaryLabel(cluster, decision)}</span>`
    + (decision.optional ? "<span class='chip optional'>◌ optional: poor example</span>" : '') + '</div>'
    + `<p class='meta'>${cluster.image_count} captures · folder <code>${cluster.source_folder}</code> · ${confidence}`
    + `${cluster.review_reasons.length ? ` · ${cluster.review_reasons.join('; ')}` : ''}`
    + `${state.viewMode === 'focus' ? ` · capture ${state.focus + 1} of ${cluster.image_count}` : ''}</p>`
    + LEGEND_STRIP;
  const members = document.createElement('div');
  members.className = 'members' + (state.viewMode === 'focus' ? ' single' : '');
  if (state.viewMode === 'focus') {
    members.appendChild(renderMemberCard(cluster, cluster.images[state.focus], state.focus));
    panel.appendChild(members);
    panel.appendChild(renderFilmstrip(cluster));
  } else {
    cluster.images.forEach((image, imageIndex) => {
      members.appendChild(renderMemberCard(cluster, image, imageIndex));
    });
    panel.appendChild(members);
  }
  host.appendChild(panel);
  renderBoxPanel(cluster);
}

function stageToImage(stage, image, clientX, clientY) {
  const picture = stage.querySelector('img');
  const rect = picture.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * image.width;
  const y = ((clientY - rect.top) / rect.height) * image.height;
  return [
    Math.max(0, Math.min(image.width, Math.round(x))),
    Math.max(0, Math.min(image.height, Math.round(y))),
  ];
}

const MINIMUM_SIDE = 6;

function resizeBox(bbox, edge, x, y, image) {
  const box = [...bbox];
  if (edge.includes('w')) box[0] = Math.min(x, box[2] - MINIMUM_SIDE);
  if (edge.includes('e')) box[2] = Math.max(x, box[0] + MINIMUM_SIDE);
  if (edge.includes('n')) box[1] = Math.min(y, box[3] - MINIMUM_SIDE);
  if (edge.includes('s')) box[3] = Math.max(y, box[1] + MINIMUM_SIDE);
  box[0] = Math.max(0, box[0]);
  box[1] = Math.max(0, box[1]);
  box[2] = Math.min(image.width, box[2]);
  box[3] = Math.min(image.height, box[3]);
  return box;
}

// Map a shape onto new bounds, so keyboard resizing keeps a trapezoid's proportions.
function fitToBounds(points, from, to) {
  const sx = (to[2] - to[0]) / Math.max(from[2] - from[0], 1);
  const sy = (to[3] - to[1]) / Math.max(from[3] - from[1], 1);
  return points.map(([x, y]) => [to[0] + (x - from[0]) * sx, to[1] + (y - from[1]) * sy]);
}

// Corner drags move one corner freely (a trapezoid). With Shift the box stays a
// rectangle: the two neighbouring corners follow along their shared sides.
function dragCorner(quad, corner, x, y, keepRect) {
  const next = quad.map((point) => [...point]);
  if (!keepRect) { next[corner] = [x, y]; return next; }
  const [x0, y0, x1, y1] = boundsOf(quad);
  const left = corner === 0 || corner === 3 ? Math.min(x, x1 - MINIMUM_SIDE) : x0;
  const right = corner === 1 || corner === 2 ? Math.max(x, x0 + MINIMUM_SIDE) : x1;
  const top = corner === 0 || corner === 1 ? Math.min(y, y1 - MINIMUM_SIDE) : y0;
  const bottom = corner === 2 || corner === 3 ? Math.max(y, y0 + MINIMUM_SIDE) : y1;
  return rectQuad([left, top, right, bottom]);
}

// Side handle i sits between corners i and i+1 (top, right, bottom, left); it moves
// both corners across the side: top and bottom vertically, right and left horizontally.
function dragSide(quad, side, dx, dy) {
  const next = quad.map((point) => [...point]);
  const shift = side % 2 === 0 ? [0, dy] : [dx, 0];
  for (const corner of [side, (side + 1) % 4]) {
    next[corner] = [next[corner][0] + shift[0], next[corner][1] + shift[1]];
  }
  return next;
}

// The magnifier: a 4x view of the image around the point being placed, with every
// box outline and a crosshair, drawn beside the cursor while a corner, side or new
// box is dragged.
const LOUPE_SIZE = 220;
const LOUPE_ZOOM = 4;

function showLoupe(stage, image, boxes, activeIndex, point, clientX, clientY) {
  const picture = stage.querySelector('img');
  if (!picture || !picture.naturalWidth) return;
  const loupe = el('loupe');
  const context = el('loupe-canvas').getContext('2d');
  const shown = picture.getBoundingClientRect().width;
  const k = picture.naturalWidth / image.width;  // source px -> preview px
  const span = (LOUPE_SIZE / LOUPE_ZOOM) * (picture.naturalWidth / shown);  // preview px across the loupe
  const scale = LOUPE_SIZE / span;
  const sx = point[0] * k - span / 2;
  const sy = point[1] * k - span / 2;
  context.fillStyle = '#111';
  context.fillRect(0, 0, LOUPE_SIZE, LOUPE_SIZE);
  context.imageSmoothingEnabled = scale < 2.5;
  context.drawImage(picture, sx, sy, span, span, 0, 0, LOUPE_SIZE, LOUPE_SIZE);
  const map = ([x, y]) => [(x * k - sx) * scale, (y * k - sy) * scale];
  boxes.forEach((box, boxIndex) => {
    context.strokeStyle = KIND_COLORS[kindOf(box)];
    context.lineWidth = boxIndex === activeIndex ? 2.5 : 1.25;
    context.beginPath();
    shapeOf(box).forEach((p, n) => { const [u, v] = map(p); if (n) context.lineTo(u, v); else context.moveTo(u, v); });
    context.closePath();
    context.stroke();
  });
  const c = LOUPE_SIZE / 2;
  context.strokeStyle = 'rgba(255,255,255,.95)';
  context.lineWidth = 1;
  context.beginPath();
  for (const [a, b] of [[[0, c], [c - 6, c]], [[c + 6, c], [LOUPE_SIZE, c]], [[c, 0], [c, c - 6]], [[c, c + 6], [c, LOUPE_SIZE]]]) {
    context.moveTo(...a); context.lineTo(...b);
  }
  context.stroke();
  context.strokeStyle = 'rgba(0,0,0,.8)';
  context.strokeRect(c - 2.5, c - 2.5, 5, 5);
  el('loupe-label').textContent = `${Math.round(point[0])}, ${Math.round(point[1])} px · ${LOUPE_ZOOM}×`;
  const gap = 28;
  let left = clientX + gap;
  if (left + LOUPE_SIZE > window.innerWidth - 8) left = clientX - gap - LOUPE_SIZE;
  let top = clientY - gap - LOUPE_SIZE;
  if (top < 8) top = clientY + gap;
  loupe.style.left = `${left}px`;
  loupe.style.top = `${top}px`;
  loupe.hidden = false;
}

function hideLoupe() {
  el('loupe').hidden = true;
}

function attachBoxEditing(stage, cluster, image, canDraw) {
  const canvas = stage.querySelector('.image-canvas');
  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    const handle = event.target.closest('.handle');
    const shapeNode = event.target.closest('.shape');
    // Outside draw mode a press on bare image only focuses the capture; boxes and handles still drag.
    if (!canDraw && !shapeNode && !handle) return;
    event.preventDefault();
    const freshDraft = !(state.draft && state.draft.clusterId === cluster.cluster_id && state.draft.imageId === image.image_id);
    const draft = openDraft(cluster, image);
    const [startX, startY] = stageToImage(stage, image, event.clientX, event.clientY);
    let mode = 'create';
    let boxIndex;
    let part = 0;
    // In draw mode a press inside a box still draws (pages cover most of the image);
    // a click there without a drag selects that box.
    const pressedIndex = shapeNode ? Number(shapeNode.dataset.boxIndex) : null;
    if (handle) {
      mode = handle.dataset.corner !== undefined ? 'corner' : 'side';
      part = Number(handle.dataset.corner ?? handle.dataset.side);
      boxIndex = Number(handle.dataset.boxIndex);
      handle.classList.add('active');
    } else if (shapeNode && !canDraw) {
      mode = 'move';
      boxIndex = pressedIndex;
    } else {
      draft.boxes.push({
        box_id: `box_${draft.boxes.length + 1}`,
        bbox: [startX, startY, startX + MINIMUM_SIDE, startY + MINIMUM_SIDE],
        polygon: null,
        kind: newBoxKind(draft.boxes),
        origin: 'reviewer',
      });
      boxIndex = draft.boxes.length - 1;
    }
    if (mode !== 'create') state.selectedBox = boxIndex;
    state.focus = Math.max(0, cluster.images.findIndex((item) => item.image_id === image.image_id));
    state.dragging = true;
    // Read the anchor from the draft at drag start, so a drag that follows an
    // earlier edit continues from the current geometry rather than a stale copy.
    const startShape = shapeOf(draft.boxes[boxIndex]).map((point) => [...point]);
    const startQuad = cornersOf(startShape);
    stage.setPointerCapture?.(event.pointerId);
    let moved = false;

    const onMove = (moveEvent) => {
      const [x, y] = stageToImage(stage, image, moveEvent.clientX, moveEvent.clientY);
      const box = draft.boxes[boxIndex];
      if (!box) return;
      if (!moved && Math.hypot(x - startX, y - startY) < 1) return;
      moved = true;
      let focusPoint = [x, y];
      if (mode === 'create') {
        box.bbox = [Math.min(startX, x), Math.min(startY, y), Math.max(startX, x), Math.max(startY, y)];
        state.selectedBox = boxIndex;
      } else if (mode === 'corner') {
        setShape(box, dragCorner(startQuad, part, x, y, moveEvent.shiftKey), image);
        box.origin = 'reviewer';
        focusPoint = box.polygon[part];
      } else if (mode === 'side') {
        setShape(box, dragSide(startQuad, part, x - startX, y - startY), image);
        box.origin = 'reviewer';
        const [a, b] = [box.polygon[part], box.polygon[(part + 1) % 4]];
        focusPoint = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      } else {
        // Moving keeps the whole outline, SAM points included.
        const [x0, y0, x1, y1] = boundsOf(startShape);
        const dx = clamp(x - startX, -x0, image.width - x1);
        const dy = clamp(y - startY, -y0, image.height - y1);
        const translated = startShape.map(([px, py]) => [px + dx, py + dy]);
        if (draft.boxes[boxIndex].polygon) setShape(box, translated, image);
        else box.bbox = boundsOf(translated);
        box.origin = 'reviewer';
      }
      renderBoxes(canvas, image, draft.boxes, true);
      canvas.querySelector(`.handle[data-box-index='${boxIndex}'][data-${mode === 'side' ? 'side' : 'corner'}='${part}']`)?.classList.add('active');
      if (mode !== 'move') showLoupe(stage, image, draft.boxes, boxIndex, focusPoint, moveEvent.clientX, moveEvent.clientY);
      renderDraftBanner();
    };

    const finish = (cancelled) => {
      stage.removeEventListener('pointermove', onMove);
      stage.removeEventListener('pointerup', onUp);
      stage.removeEventListener('pointercancel', onCancel);
      hideLoupe();
      state.dragging = false;
      if (cancelled) { cancelDraft(); return; }
      const box = draft.boxes[boxIndex];
      if (mode === 'create' && (!moved || box.bbox[2] - box.bbox[0] < MINIMUM_SIDE * 2 || box.bbox[3] - box.bbox[1] < MINIMUM_SIDE * 2)) {
        draft.boxes.splice(boxIndex, 1);
        state.selectedBox = moved ? null : pressedIndex;
        if (!moved) {
          if (freshDraft) state.draft = null;
          render();
          return;
        }
      }
      // A click without a drag only selects: it must not save an unchanged box set as an edit.
      if (!moved && mode !== 'create') {
        if (freshDraft) { state.draft = null; }
        render();
        return;
      }
      commitDraft();
    };
    const onUp = () => finish(false);
    const onCancel = () => finish(true);
    stage.addEventListener('pointermove', onMove);
    stage.addEventListener('pointerup', onUp);
    stage.addEventListener('pointercancel', onCancel);
  });
}

async function setMembership(cluster, image, included) {
  if (hasDraft()) await commitDraft();
  pushUndo(cluster.cluster_id);
  await api(`/api/clusters/${encodeURIComponent(cluster.cluster_id)}/images/${encodeURIComponent(image.image_id)}/membership`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({included}),
  });
  render();
}

async function approveBoxes(cluster, image, approved) {
  if (hasDraft()) await commitDraft();
  pushUndo(cluster.cluster_id);
  const payload = await api(`/api/clusters/${encodeURIComponent(cluster.cluster_id)}/images/${encodeURIComponent(image.image_id)}/bbox-status`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({approved}),
  });
  applyImage(cluster.cluster_id, image.image_id, payload.image);
  render();
}

async function clusterAction(action) {
  const cluster = currentCluster();
  if (!cluster) return;
  if (hasDraft()) await commitDraft();
  pushUndo(cluster.cluster_id);
  await api(`/api/clusters/${encodeURIComponent(cluster.cluster_id)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action}),
  });
  if (action === 'reopen') render();
  else step(1);
}

const RELATION_LABELS = {
  same_document: 'same document',
  same_background_document: 'same background document',
  different_document: 'different document',
  unsure: 'unsure',
};

function primaryLabel(cluster, decision) {
  if (decision.primary_view === 'none') return 'no unoccluded capture';
  const index = cluster.images.findIndex((image) => image.image_id === decision.primary_view);
  return index >= 0 ? `primary view: capture ${index + 1}` : 'primary view not set';
}

async function postClusterDecision(clusterId, body) {
  if (hasDraft()) await commitDraft();
  pushUndo(clusterId);
  return api(`/api/clusters/${encodeURIComponent(clusterId)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
}

// Same-* relations stay on the cluster so the primary view can be marked next;
// different/unsure need no primary view and advance at once.
async function setRelation(relation) {
  const cluster = currentCluster();
  if (!cluster) return;
  const payload = await postClusterDecision(cluster.cluster_id, {action: 'relation', relation});
  if (relation === 'different_document' || relation === 'unsure' || payload.cluster.primary_view) step(1);
  else render();
}

// Marking the primary view finishes a pair once its relation is set, so advance.
async function setPrimaryView(cluster, primaryView) {
  const payload = await postClusterDecision(cluster.cluster_id, {action: 'primary_view', primary_view: primaryView});
  if (primaryView && payload.cluster.relation) step(1);
  else render();
}

function primaryFocused() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (!cluster || !image) return;
  const current = decisionFor(cluster.cluster_id).primary_view;
  setPrimaryView(cluster, current === image.image_id ? null : image.image_id);
}

function step(delta) {
  if (!state.queue.length) return;
  if (hasDraft()) commitDraft();
  state.index = Math.max(0, Math.min(state.queue.length - 1, state.index + delta));
  state.focus = 0;
  state.selectedBox = null;
  render();
  window.scrollTo({top: 0, behavior: 'instant'});
}

function flipCapture(delta) {
  const cluster = currentCluster();
  if (!cluster) return;
  if (hasDraft()) commitDraft();
  state.focus = Math.max(0, Math.min(cluster.images.length - 1, state.focus + delta));
  state.selectedBox = null;
  render();
}

function toggleViewMode() {
  if (hasDraft()) commitDraft();
  state.viewMode = state.viewMode === 'grid' ? 'focus' : 'grid';
  render();
}

function toggleFocused() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (cluster && image) setMembership(cluster, image, isExcluded(cluster.cluster_id, image.image_id));
}

function toggleByNumber(number) {
  const cluster = currentCluster();
  const image = cluster?.images[number - 1];
  if (!cluster || !image) return;
  state.focus = number - 1;
  setMembership(cluster, image, isExcluded(cluster.cluster_id, image.image_id));
}

function selectNextBox() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (!cluster || !image) return;
  const count = currentBoxes(cluster, image).length;
  if (!count) return;
  state.selectedBox = state.selectedBox === null ? 0 : (state.selectedBox + 1) % count;
  render();
}

function deleteSelectedBox() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (!cluster || !image || state.selectedBox === null) return;
  const draft = openDraft(cluster, image);
  if (state.selectedBox >= draft.boxes.length) return;
  draft.boxes.splice(state.selectedBox, 1);
  state.selectedBox = null;
  commitDraft();
}

function nudgeSelectedBox(dx, dy, resize) {
  const cluster = currentCluster();
  const image = focusedImage();
  if (!cluster || !image || state.selectedBox === null) return;
  const draft = openDraft(cluster, image);
  const box = draft.boxes[state.selectedBox];
  if (!box) return;
  const stepSize = 8;
  const old = [...box.bbox];
  if (resize) {
    box.bbox = resizeBox(box.bbox, 'se', box.bbox[2] + dx * stepSize, box.bbox[3] + dy * stepSize, image);
  } else {
    const width = box.bbox[2] - box.bbox[0];
    const height = box.bbox[3] - box.bbox[1];
    const left = Math.max(0, Math.min(image.width - width, box.bbox[0] + dx * stepSize));
    const top = Math.max(0, Math.min(image.height - height, box.bbox[1] + dy * stepSize));
    box.bbox = [left, top, left + width, top + height];
  }
  if (box.polygon) setShape(box, fitToBounds(box.polygon, old, box.bbox), image);
  box.origin = 'reviewer';
  render();
  scheduleCommit(700);
}

// Optional = technically right but a poor example (a blank sheet on a blank page):
// credit if a model finds it, no penalty if it does not. Never advances.
async function toggleClusterOptional() {
  const cluster = currentCluster();
  if (!cluster) return;
  await postClusterDecision(cluster.cluster_id, {action: 'optional', optional: !decisionFor(cluster.cluster_id).optional});
  render();
}

function toggleSelectedBoxOptional() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (!cluster || !image || state.selectedBox === null || !currentBoxes(cluster, image)[state.selectedBox]) return;
  const box = openDraft(cluster, image).boxes[state.selectedBox];
  box.optional = !box.optional;
  commitDraft();
}

function squareUpSelectedBox() {
  const cluster = currentCluster();
  const image = focusedImage();
  if (!cluster || !image || state.selectedBox === null) return;
  if (!currentBoxes(cluster, image)[state.selectedBox]?.polygon) return;
  const box = openDraft(cluster, image).boxes[state.selectedBox];
  box.polygon = null;
  box.origin = 'reviewer';
  commitDraft();
}

function exitEditing() {
  if (cancelDraft()) return;
  if (state.selectedBox !== null) { state.selectedBox = null; render(); return; }
  if (state.editing) { state.editing = false; render(); }
}

async function markRemaining(scope) {
  const label = scope === 'bboxes' ? 'box sets' : 'clusters';
  if (!window.confirm(`Approve every unreviewed ${label} in the whole dataset?`)) return;
  await api('/api/mark-remaining-ok', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({scope}),
  });
  await loadDecisions();
  render();
}

async function exportReview() {
  if (hasDraft()) await commitDraft();
  const payload = await api('/api/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  setSave('ok', `Exported ${payload.final_cluster_count} clusters to ${payload.artifacts.clusters_reviewed}`);
}

async function loadDecisions() {
  const response = await fetch('/api/decisions');
  const payload = await response.json();
  state.decisions = payload.clusters || {};
  state.progress = payload.progress;
  el('decisions-path').textContent = payload.decisions_path;
}

function bindControls() {
  el('prev').addEventListener('click', () => step(-1));
  el('next').addEventListener('click', () => step(1));
  el('approve').addEventListener('click', () => clusterAction('approve'));
  el('dissolve').addEventListener('click', () => clusterAction('dissolve'));
  el('mark-irregular').addEventListener('click', () => clusterAction('irregular'));
  el('reopen').addEventListener('click', () => clusterAction('reopen'));
  el('rel-same').addEventListener('click', () => setRelation('same_document'));
  el('rel-background').addEventListener('click', () => setRelation('same_background_document'));
  el('rel-different').addEventListener('click', () => setRelation('different_document'));
  el('rel-unsure').addEventListener('click', () => setRelation('unsure'));
  el('primary-focused').addEventListener('click', primaryFocused);
  el('primary-none').addEventListener('click', () => {
    const cluster = currentCluster();
    if (cluster) setPrimaryView(cluster, 'none');
  });
  el('toggle-focused').addEventListener('click', toggleFocused);
  el('view-mode').addEventListener('click', toggleViewMode);
  el('edit-mode').addEventListener('click', () => {
    if (state.editing) exitEditing();
    else { state.editing = true; render(); }
  });
  el('approve-boxes').addEventListener('click', () => {
    const cluster = currentCluster();
    const image = focusedImage();
    if (cluster && image) approveBoxes(cluster, image, imageDecision(cluster.cluster_id, image.image_id).bbox_status !== 'approved');
  });
  el('select-box').addEventListener('click', selectNextBox);
  el('delete-box').addEventListener('click', deleteSelectedBox);
  el('box-kind').addEventListener('click', toggleSelectedBoxKind);
  el('square-box').addEventListener('click', squareUpSelectedBox);
  el('box-optional').addEventListener('click', toggleSelectedBoxOptional);
  el('optional').addEventListener('click', toggleClusterOptional);
  el('undo').addEventListener('click', undo);
  el('draft-save').addEventListener('click', () => commitDraft());
  el('draft-cancel').addEventListener('click', () => cancelDraft());
  el('mark-remaining').addEventListener('click', () => markRemaining('clusters'));
  el('mark-remaining-boxes').addEventListener('click', () => markRemaining('bboxes'));
  el('export').addEventListener('click', exportReview);
  for (const id of ['min-size', 'sort', 'status-filter']) {
    el(id).addEventListener('change', () => { state.index = 0; render(); });
  }
  el('search').addEventListener('input', () => { state.index = 0; render(); });
}

function isTyping(event) {
  const tag = event.target.tagName;
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
}

function bindKeyboard() {
  document.addEventListener('keydown', (event) => {
    if (isTyping(event)) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      undo();
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const key = event.key;
    if (key >= '1' && key <= '9') { event.preventDefault(); toggleByNumber(Number(key)); return; }
    switch (key) {
      case 'j': case 'J': event.preventDefault(); step(1); break;
      case 'k': case 'K': event.preventDefault(); step(-1); break;
      case 'v': case 'V': event.preventDefault(); toggleViewMode(); break;
      case 'a': case 'A': event.preventDefault(); clusterAction('approve'); break;
      case 'd': case 'D': event.preventDefault(); clusterAction('dissolve'); break;
      case 'i': case 'I': event.preventDefault(); clusterAction('irregular'); break;
      case 'r': case 'R': event.preventDefault(); clusterAction('reopen'); break;
      case 's': case 'S': event.preventDefault(); setRelation('same_document'); break;
      case 'g': case 'G': event.preventDefault(); setRelation('same_background_document'); break;
      case 'n': case 'N': event.preventDefault(); setRelation('different_document'); break;
      case 'u': case 'U': event.preventDefault(); setRelation('unsure'); break;
      case 'p': case 'P': event.preventDefault(); primaryFocused(); break;
      case 'o': case 'O': {
        event.preventDefault();
        const cluster = currentCluster();
        if (cluster) setPrimaryView(cluster, 'none');
        break;
      }
      case 'x': case 'X': event.preventDefault(); toggleFocused(); break;
      case 'e': case 'E':
        event.preventDefault();
        if (state.editing) exitEditing();
        else { state.editing = true; render(); }
        break;
      case 'b': case 'B': {
        event.preventDefault();
        const cluster = currentCluster();
        const image = focusedImage();
        if (cluster && image) approveBoxes(cluster, image, imageDecision(cluster.cluster_id, image.image_id).bbox_status !== 'approved');
        break;
      }
      case 'Tab': event.preventDefault(); selectNextBox(); break;
      case 't': case 'T': event.preventDefault(); toggleSelectedBoxKind(); break;
      case 'q': case 'Q': event.preventDefault(); squareUpSelectedBox(); break;
      case 'l': case 'L': event.preventDefault(); toggleClusterOptional(); break;
      case 'w': case 'W': event.preventDefault(); toggleSelectedBoxOptional(); break;
      case 'Enter': event.preventDefault(); if (hasDraft()) commitDraft(); break;
      case 'Escape': event.preventDefault(); exitEditing(); break;
      case 'Delete': case 'Backspace': event.preventDefault(); deleteSelectedBox(); break;
      case 'ArrowRight':
        event.preventDefault();
        if (state.selectedBox !== null) nudgeSelectedBox(1, 0, event.shiftKey);
        else flipCapture(1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        if (state.selectedBox !== null) nudgeSelectedBox(-1, 0, event.shiftKey);
        else flipCapture(-1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        if (state.selectedBox !== null) nudgeSelectedBox(0, 1, event.shiftKey);
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (state.selectedBox !== null) nudgeSelectedBox(0, -1, event.shiftKey);
        break;
      default: break;
    }
  });
  window.addEventListener('beforeunload', (event) => {
    if (!hasDraft()) return;
    event.preventDefault();
    event.returnValue = '';
  });
}

async function boot() {
  const response = await fetch('/api/dataset');
  state.dataset = await response.json();
  el('min-size').value = String(state.dataset.defaults.minimum_cluster_size);
  el('sort').value = state.dataset.defaults.sort;
  await loadDecisions();
  bindControls();
  bindKeyboard();
  setSave('ok', 'Every change autosaves to disk');
  render();
}

boot();
"""

BODY = """
<header class='top'>
  <h1>Cluster review</h1>
  <div class='nav'>
    <button type='button' id='prev' title='Previous cluster (key k)'>◀ <kbd>k</kbd></button>
    <span id='queue-position'></span>
    <button type='button' id='next' title='Next cluster (key j)'><kbd>j</kbd> ▶</button>
  </div>
  <details class='filters'><summary>Filter &amp; sort ▾</summary>
    <div class='filter-pop'>
      <label for='search'>Search</label><input id='search' type='search' placeholder='cluster, folder, reason'>
      <label for='min-size'>Minimum size</label><select id='min-size'>
        <option value='1'>all clusters</option><option value='2'>2+ images</option>
        <option value='3'>3+ images</option><option value='4'>4+ images</option>
        <option value='10'>10+ images</option></select>
      <label for='sort'>Sort</label><select id='sort'>
        <option value='confidence-asc'>weakest confidence first</option>
        <option value='confidence-desc'>strongest confidence first</option>
        <option value='size-desc'>largest first</option><option value='size-asc'>smallest first</option>
        <option value='folder'>source folder</option><option value='id'>cluster ID</option></select>
      <label for='status-filter'>Status</label><select id='status-filter'>
        <option value=''>all</option><option value='unreviewed'>unreviewed</option>
        <option value='reviewed'>reviewed</option><option value='approved'>approved</option>
        <option value='edited'>edited</option><option value='dissolved'>dissolved</option>
        <option value='irregular'>irregular</option>
        <option value='needs-relation'>confirmed, relation not set</option>
        <option value='needs-primary'>confirmed, primary view not set</option></select>
    </div>
  </details>
  <button type='button' id='view-mode' aria-pressed='false' title='Switch between side-by-side and full-screen single-capture view (key v)'>View: <span id='view-mode-label'>Side by side</span> <kbd>v</kbd></button>
  <span class='spacer'></span>
  <button type='button' id='undo' title='Undo the last saved change (Ctrl+Z)'>Undo <kbd>Ctrl</kbd><kbd>Z</kbd></button>
  <span id='save-state' class='save-state ok' role='status' aria-live='polite'>Loading</span>
</header>
<main id='main'>
  <section class='workspace'>
    <div id='draft-banner' class='draft-banner' role='status' aria-live='polite' hidden>
      <span id='draft-summary'></span>
      <button type='button' id='draft-save' class='primary' title='Save this box edit now (Enter)'>Save boxes <kbd>Enter</kbd></button>
      <button type='button' id='draft-cancel' title='Discard this box edit (Escape)'>Cancel <kbd>Esc</kbd></button>
      <span>Releasing the mouse saves automatically.</span>
    </div>
    <p class='progress' id='progress'></p>
    <div id='cluster-host'></div>
  </section>
  <aside>
    <section class='card'>
      <h3>1 · Are these the same document?</h3>
      <div class='relations'>
        <button type='button' id='rel-same' title='Literally the same document: same visible sheet, re-captured or with/without a cover (key s); approves the grouping'><b>Same document <kbd>s</kbd></b><small>same sheet, maybe uncovered</small></button>
        <button type='button' id='rel-background' title='Same underlying base page, but a different foreground sheet on top (key g); approves the grouping'><b>Same background <kbd>g</kbd></b><small>same page, different sheet on top</small></button>
        <button type='button' id='rel-different' title='Different documents (key n); dissolves the grouping and advances'><b>Different <kbd>n</kbd></b><small>dissolves, advances</small></button>
        <button type='button' id='rel-unsure' title='Cannot tell (key u); marks irregular and advances'><b>Unsure <kbd>u</kbd></b><small>irregular, advances</small></button>
      </div>
      <button type='button' id='optional' class='optional-toggle' aria-pressed='false' title='Technically right but a poor example, e.g. a blank sheet on a blank page: a model gets credit for finding it and no penalty for missing it (key l)'><b>◌ Optional: poor example <kbd>l</kbd></b><small>right, but e.g. blank on blank; no penalty if a model misses it</small></button>
      <h3 style='margin-top:.8rem'>2 · Which capture is unoccluded?</h3>
      <div class='row'>
        <button type='button' id='primary-focused' title='Mark the focused capture as the primary unoccluded view; press again to clear (key p)'>★ Focused capture <kbd>p</kbd></button>
        <button type='button' id='primary-none' title='No capture shows the document unoccluded (key o)'>None <kbd>o</kbd></button>
      </div>
      <h3 style='margin-top:.8rem'>Grouping</h3>
      <div class='row'>
        <button type='button' id='approve' title='Mark this grouping correct and advance (key a)'>Mark OK <kbd>a</kbd></button>
        <button type='button' id='dissolve' title='Reject the whole grouping and advance (key d)'>Dissolve <kbd>d</kbd></button>
        <button type='button' id='mark-irregular' class='irregular' title='Mark this grouping irregular and exclude its images from training and validation (key i)'>Irregular <kbd>i</kbd></button>
        <button type='button' id='reopen' title='Return this cluster to unreviewed (key r)'>Reopen <kbd>r</kbd></button>
        <button type='button' id='toggle-focused' title='Include or exclude the focused capture (key x, or its number key)'>Toggle capture <kbd>x</kbd></button>
      </div>
    </section>
    <section class='card'>
      <h3 id='box-panel-title'>3 · Boxes</h3>
      <div id='box-list' class='box-list'></div>
      <div class='row'>
        <button type='button' id='edit-mode' aria-pressed='false' title='Draw mode: drag on the image to add a box (key e)'>✚ Draw box <kbd>e</kbd></button>
        <button type='button' id='select-box' title='Select the next box on the focused capture (Tab)'>Next <kbd>Tab</kbd></button>
        <button type='button' id='box-kind' title='Selected box: page of interest ↔ occluder (key t)'>Page ↔ occluder <kbd>t</kbd></button>
        <button type='button' id='square-box' title='Turn the selected box back into an upright rectangle (key q)'>Make rectangle <kbd>q</kbd></button>
        <button type='button' id='box-optional' aria-pressed='false' title='Selected box is optional: real but a poor example (blank on blank); no penalty when missed (key w)'>Optional box <kbd>w</kbd></button>
        <button type='button' id='delete-box' class='danger' title='Delete the selected box (Delete)'>Delete <kbd>Del</kbd></button>
        <button type='button' id='approve-boxes' class='primary' title='Approve the boxes on the focused capture (key b)'>Approve boxes <kbd>b</kbd></button>
      </div>
      <p class='hint'>Corner alone = trapezoid; <kbd>Shift</kbd> keeps a rectangle. Square handle moves a side; inside moves the box. A magnifier shows while you drag. <kbd>←</kbd> <kbd>→</kbd> switch capture.</p>
    </section>
    <details class='card' id='legend-card'>
      <summary>What the outlines mean</summary>
      <div class='legend'>
        <svg width='44' height='22'><rect x='2' y='2' width='40' height='18' rx='1' fill='#2fb344' fill-opacity='.15' stroke='#2fb344' stroke-width='2'/></svg>
        <span><b style='color:#2a9a3c'>Green</b>: the page of interest (the document being matched)</span>
        <svg width='44' height='22'><rect x='2' y='2' width='40' height='18' rx='1' fill='#f03e3e' fill-opacity='.15' stroke='#f03e3e' stroke-width='2'/></svg>
        <span><b style='color:#d32f2f'>Red</b>: an occluder, a sheet lying on top of the page</span>
        <svg width='44' height='22'><rect x='2' y='2' width='40' height='18' rx='1' fill='#748ffc' fill-opacity='.15' stroke='#748ffc' stroke-width='2'/></svg>
        <span><b style='color:#4c6ef5'>Blue</b>: unlabelled; press <kbd>t</kbd> to make it page or occluder</span>
        <svg width='44' height='22'><polygon points='6,3 40,2 42,20 2,19' fill='#2fb344' fill-opacity='.15' stroke='#2fb344' stroke-width='2'/></svg>
        <span><b>Solid outline</b>: the box's real shape (rectangle, 4-corner trapezoid, or the SAM outline). Exported as the polygon.</span>
        <svg width='44' height='22'><rect x='2' y='2' width='40' height='18' fill='none' stroke='#2fb344' stroke-width='1.2' stroke-dasharray='5 4'/><polygon points='9,3 38,2 42,20 2,19' fill='none' stroke='#2fb344' stroke-width='2'/></svg>
        <span><b>Dashed rectangle</b>: the upright bounding box of a tilted shape. Exported as the bbox; you don't edit it directly.</span>
        <svg width='44' height='22'><rect x='2' y='2' width='40' height='18' fill='#f03e3e' fill-opacity='.25' stroke='#f03e3e' stroke-width='3.5'/></svg>
        <span><b>Thick outline</b>: the selected box (<kbd>Tab</kbd> cycles). Round handles are corners; square ones are sides.</span>
        <svg width='44' height='22'><rect x='2' y='2' width='40' height='18' fill='#f03e3e' fill-opacity='.05' stroke='#f03e3e' stroke-width='2' stroke-dasharray='1.5 3.5' stroke-linecap='round'/></svg>
        <span><b>Dotted outline</b>: optional, a real box that is a poor example (blank on blank). Missing it costs a model nothing.</span>
      </div>
      <p class='hint'>The tag on each box gives its number, kind and source: <i>SAM via cropper/diff/gemini</i> is a machine proposal with its confidence; <i>your edit</i> means you changed it.</p>
    </details>
    <details class='card'>
      <summary>All keyboard shortcuts</summary>
      <table>
        <tr><td><kbd>j</kbd> / <kbd>k</kbd></td><td>next / previous cluster</td></tr>
        <tr><td><kbd>v</kbd></td><td>side by side ↔ full screen</td></tr>
        <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>flip between captures</td></tr>
        <tr><td><kbd>1</kbd>–<kbd>9</kbd></td><td>toggle that capture</td></tr>
        <tr><td><kbd>x</kbd></td><td>toggle focused capture</td></tr>
        <tr><td><kbd>a</kbd></td><td>mark cluster OK, advance</td></tr>
        <tr><td><kbd>d</kbd></td><td>dissolve cluster, advance</td></tr>
        <tr><td><kbd>i</kbd></td><td>mark irregular; exclude from training/validation, advance</td></tr>
        <tr><td><kbd>r</kbd></td><td>reopen as unreviewed</td></tr>
        <tr><td><kbd>s</kbd></td><td>same document (literally the same sheet)</td></tr>
        <tr><td><kbd>g</kbd></td><td>same background document, different foreground sheet</td></tr>
        <tr><td><kbd>n</kbd></td><td>different documents, advance</td></tr>
        <tr><td><kbd>u</kbd></td><td>unsure, advance</td></tr>
        <tr><td><kbd>p</kbd></td><td>focused capture is the primary unoccluded view (advances once relation is set)</td></tr>
        <tr><td><kbd>o</kbd></td><td>no capture is unoccluded</td></tr>
        <tr><td><kbd>Ctrl</kbd>+<kbd>Z</kbd></td><td>undo last change</td></tr>
        <tr><td><kbd>e</kbd></td><td>draw mode (drag on the image to add a box); existing boxes are always draggable by their handles</td></tr>
        <tr><td><kbd>Tab</kbd></td><td>select next box</td></tr>
        <tr><td><kbd>b</kbd></td><td>approve boxes on capture</td></tr>
        <tr><td><kbd>t</kbd></td><td>selected box: page of interest ↔ occluder (green / red)</td></tr>
        <tr><td><kbd>q</kbd></td><td>selected box back to an upright rectangle</td></tr>
        <tr><td><kbd>l</kbd></td><td>cluster is optional: right but a poor example (no penalty if missed)</td></tr>
        <tr><td><kbd>w</kbd></td><td>selected box is optional (dotted)</td></tr>
        <tr><td><kbd>Del</kbd></td><td>delete selected box</td></tr>
        <tr><td>arrows</td><td>nudge selected box</td></tr>
        <tr><td><kbd>Shift</kbd>+arrows</td><td>resize selected box</td></tr>
        <tr><td><kbd>Shift</kbd>+corner drag</td><td>keep the box a rectangle</td></tr>
        <tr><td><kbd>Enter</kbd></td><td>save a pending box edit</td></tr>
        <tr><td><kbd>Esc</kbd></td><td>cancel edit / clear selection</td></tr>
      </table>
    </details>
    <details class='card'>
      <summary>Finish the pass</summary>
      <div class='row'>
        <button type='button' id='mark-remaining' title='Approve every cluster still unreviewed'>Mark remaining clusters OK</button>
        <button type='button' id='mark-remaining-boxes' title='Approve every box set still unreviewed'>Mark remaining boxes OK</button>
        <button type='button' id='export' title='Write corrected cluster and box manifests'>Export corrected manifests</button>
      </div>
      <p class='hint'>Dropping a 2-image cluster to one capture dissolves it automatically. Every change autosaves to <code id='decisions-path'></code></p>
    </details>
  </aside>
</main>
<div id='loupe' hidden><canvas id='loupe-canvas' width='220' height='220'></canvas><span id='loupe-label'></span></div>
"""


def review_app_html() -> str:
    """Return the complete review client document."""
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Cluster review</title>"
        f"<style>{STYLE}</style></head><body>{BODY}"
        f"<script>{SCRIPT}</script></body></html>"
    )
