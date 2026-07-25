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
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,sans-serif;background:#eef1f5;color:#16202b}
header.bar{position:sticky;top:0;z-index:6;display:flex;flex-wrap:wrap;gap:.75rem;align-items:center;
  padding:.6rem .9rem;background:#132132;color:#f4f7fa;box-shadow:0 2px 8px rgba(0,0,0,.25)}
header.bar h1{font-size:1rem;margin:0 1rem 0 0;white-space:nowrap}
.controls{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center}
button{font:inherit;padding:.4rem .6rem;border:1px solid #7d93ab;border-radius:.3rem;background:#eaf1f8;
  color:#16202b;cursor:pointer}
button:hover{background:#dbe7f3}
button:focus-visible{outline:3px solid #ffd166;outline-offset:2px}
button[aria-pressed=true]{background:#1f6feb;border-color:#1f6feb;color:#fff}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:#1f6feb;border-color:#1f6feb;color:#fff;font-weight:700}
button.danger{background:#fbdcdc;border-color:#c0504d}
kbd{font:inherit;font-size:.75rem;padding:0 .25rem;border:1px solid #97a9bd;border-radius:.2rem;background:#fff;color:#16202b}
select,input[type=search]{font:inherit;padding:.35rem;border:1px solid #7d93ab;border-radius:.3rem}
label.field{display:flex;flex-direction:column;gap:.15rem;font-size:.75rem;font-weight:600;color:#c9d6e4}
.progress{font-size:.82rem;color:#425464;margin:.2rem 0 .6rem}
.save-state{font-size:.82rem;font-weight:700}
.save-state.ok{color:#7ee2a8}
.save-state.pending{color:#ffd166}
.save-state.error{color:#ff9a9a}
main{padding:1rem;display:grid;grid-template-columns:minmax(0,1fr) 17rem;gap:1rem;align-items:start}
main.focus-mode{grid-template-columns:minmax(0,1fr)}
main.focus-mode aside{display:none}
.cluster{background:#fff;border:1px solid #c3cedb;border-radius:.4rem;padding:1rem}
.cluster h2{margin:0 0 .2rem;font-size:1.1rem}
.cluster .meta{margin:0 0 .6rem;font-size:.85rem;color:#425464}
.status-chip{display:inline-block;padding:.1rem .5rem;border-radius:1rem;font-size:.75rem;font-weight:700;
  border:1px solid #97a9bd;background:#f1f5f9}
.status-chip.approved{background:#d8f5e3;border-color:#3fa96c}
.status-chip.edited{background:#fdf0cf;border-color:#d8a72a}
.status-chip.dissolved{background:#fbdcdc;border-color:#c0504d}
.status-chip.irregular{background:#eee3ff;border-color:#7950f2}
.members{display:grid;grid-template-columns:repeat(auto-fit,minmax(19rem,1fr));gap:.8rem;margin-top:.6rem}
.members.single{grid-template-columns:minmax(0,1fr)}
.member{border:2px solid #cbd5e1;border-radius:.35rem;padding:.5rem;background:#f8fafc}
.member.focused{border-color:#1f6feb;box-shadow:0 0 0 2px rgba(31,111,235,.25)}
.member.excluded{background:#f3e6e6;border-color:#c0504d}
.member.excluded .stage{opacity:.45}
.member-head{display:flex;justify-content:space-between;gap:.4rem;align-items:baseline;font-size:.8rem;font-weight:700}
.member-head .name{overflow-wrap:anywhere;font-weight:600}
.stage{position:relative;display:flex;justify-content:center;align-items:flex-start;margin:.4rem 0;background:#dde4ec;-webkit-user-select:none;user-select:none;touch-action:none}
.image-canvas{position:relative;width:100%;line-height:0}
.stage img{display:block;width:100%;height:auto}
.focus-mode .image-canvas{width:fit-content;max-width:100%}
.focus-mode .stage img{max-height:calc(100vh - 15rem);max-width:100%;width:auto;margin:0}
.stage.editing{cursor:crosshair}
.box{position:absolute;border:2px solid #1f6feb;background:rgba(31,111,235,.12)}
.box.editable{cursor:move}
.box.selected{border-color:#e8590c;background:rgba(232,89,12,.16);z-index:2}
.box.draft{border-style:dashed}
.box .tag{position:absolute;top:-1.15rem;left:0;font-size:.66rem;background:#132132;color:#fff;padding:0 .25rem;white-space:nowrap}
.handle{position:absolute;width:12px;height:12px;background:#fff;border:2px solid #e8590c;border-radius:2px}
.handle[data-edge=nw]{left:-6px;top:-6px;cursor:nwse-resize}
.handle[data-edge=ne]{right:-6px;top:-6px;cursor:nesw-resize}
.handle[data-edge=sw]{left:-6px;bottom:-6px;cursor:nesw-resize}
.handle[data-edge=se]{right:-6px;bottom:-6px;cursor:nwse-resize}
.handle[data-edge=n]{left:50%;top:-6px;margin-left:-6px;cursor:ns-resize}
.handle[data-edge=s]{left:50%;bottom:-6px;margin-left:-6px;cursor:ns-resize}
.handle[data-edge=w]{left:-6px;top:50%;margin-top:-6px;cursor:ew-resize}
.handle[data-edge=e]{right:-6px;top:50%;margin-top:-6px;cursor:ew-resize}
.member-foot{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;font-size:.75rem}
.bbox-chip{padding:.1rem .4rem;border-radius:1rem;border:1px solid #97a9bd;background:#fff;font-weight:700}
.bbox-chip.approved{background:#d8f5e3;border-color:#3fa96c}
.bbox-chip.edited{background:#fdf0cf;border-color:#d8a72a}
.filmstrip{display:flex;gap:.4rem;overflow-x:auto;padding:.4rem 0;margin-top:.5rem}
.filmstrip button{padding:.2rem;border-width:2px;flex:0 0 auto}
.filmstrip button[aria-current=true]{border-color:#1f6feb}
.filmstrip img{display:block;height:4.5rem;width:auto}
.filmstrip .excluded-thumb{opacity:.4}
.draft-banner{position:sticky;top:3.4rem;z-index:5;display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;
  height:3.4rem;overflow:auto;padding:.5rem .7rem;margin:0 0 .6rem;background:#fff4d1;border:2px solid #d8a72a;border-radius:.35rem;font-size:.85rem;font-weight:600}
/* Keep the draft slot in normal flow while hiding its contents. Showing or hiding
   the save controls must never move the progress line or image cards. */
.draft-banner[hidden]{display:flex !important;visibility:hidden;pointer-events:none}
aside{background:#fff;border:1px solid #c3cedb;border-radius:.4rem;padding:.8rem;position:sticky;top:4.2rem}
aside h3{margin:.2rem 0 .4rem;font-size:.9rem}
aside table{width:100%;border-collapse:collapse;font-size:.78rem}
aside td{padding:.12rem 0;vertical-align:top}
aside td:first-child{white-space:nowrap;padding-right:.5rem}
.hint{font-size:.78rem;color:#425464;margin:.5rem 0 0}
.empty{padding:2rem;text-align:center;color:#425464}
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
  return {status: 'unreviewed', dissolved: false, excluded_image_ids: [], images: {}};
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
  return boxes.map((box) => ({...box, bbox: [...box.bbox]}));
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
  const boxes = draft.boxes.map((box) => ({...box, bbox: box.bbox.map((value) => Math.round(value))}));
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
    const statusMatch = !statusFilter
      || (statusFilter === 'unreviewed' ? decision.status === 'unreviewed'
        : statusFilter === 'reviewed' ? decision.status !== 'unreviewed'
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
  if (!p) return;
  el('progress').textContent =
    `${p.reviewed_cluster_count} of ${p.cluster_count} clusters reviewed `
    + `(${p.approved_cluster_count} approved, ${p.edited_cluster_count} edited, `
    + `${p.dissolved_cluster_count} dissolved, ${p.irregular_cluster_count} irregular); `
    + `${p.bbox_approved_image_count} of ${p.image_count} images with approved boxes`;
  el('queue-position').textContent = state.queue.length
    ? `Cluster ${state.index + 1} of ${state.queue.length} in queue`
    : 'No clusters match the filters';
}

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

function renderBoxes(stage, image, boxes, editable) {
  for (const node of [...stage.querySelectorAll('.box')]) node.remove();
  boxes.forEach((box, boxIndex) => {
    const [x0, y0, x1, y1] = box.bbox;
    const node = document.createElement('div');
    const selected = editable && state.selectedBox === boxIndex;
    node.className = 'box' + (editable ? ' editable' : '') + (selected ? ' selected' : '')
      + (hasDraft() ? ' draft' : '');
    node.style.left = `${(x0 / image.width) * 100}%`;
    node.style.top = `${(y0 / image.height) * 100}%`;
    node.style.width = `${((x1 - x0) / image.width) * 100}%`;
    node.style.height = `${((y1 - y0) / image.height) * 100}%`;
    node.dataset.boxIndex = String(boxIndex);
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = `${boxIndex + 1} ${box.kind || 'box'} ${Math.round(x1 - x0)}×${Math.round(y1 - y0)}`;
    node.appendChild(tag);
    if (selected) {
      for (const edge of HANDLES) {
        const handle = document.createElement('span');
        handle.className = 'handle';
        handle.dataset.edge = edge;
        handle.dataset.boxIndex = String(boxIndex);
        node.appendChild(handle);
      }
    }
    stage.appendChild(node);
  });
}

function renderDraftBanner() {
  const banner = el('draft-banner');
  banner.hidden = !hasDraft();
  if (hasDraft()) {
    el('draft-summary').textContent =
      `Unsaved box edit on ${state.draft.imageId.split('/').pop()} (${state.draft.boxes.length} boxes)`;
  }
}

function renderMemberCard(cluster, image, imageIndex) {
  const excluded = isExcluded(cluster.cluster_id, image.image_id);
  const imageState = imageDecision(cluster.cluster_id, image.image_id);
  const boxes = currentBoxes(cluster, image);
  const isFocused = imageIndex === state.focus;
  const editable = state.editing && isFocused;
  const card = document.createElement('article');
  card.className = 'member' + (isFocused ? ' focused' : '') + (excluded ? ' excluded' : '');
  card.innerHTML =
    `<div class='member-head'><span><kbd>${imageIndex + 1}</kbd> #${image.sequence_index}</span>`
    + `<span class='name'>${image.filename}</span></div>`
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
  if (image.width && image.height) renderBoxes(canvas, image, boxes, editable);
  const foot = card.querySelector('.member-foot');
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.setAttribute('aria-pressed', String(!excluded));
  toggle.title = `Include this capture in the cluster (key ${imageIndex + 1}, or x when focused)`;
  toggle.textContent = excluded ? 'Excluded' : 'In cluster';
  toggle.addEventListener('click', () => setMembership(cluster, image, excluded));
  foot.appendChild(toggle);
  const chip = document.createElement('span');
  chip.className = `bbox-chip ${imageState.bbox_status}`;
  chip.textContent = `boxes ${boxes.length} · ${imageState.bbox_status}`;
  foot.appendChild(chip);
  const approve = document.createElement('button');
  approve.type = 'button';
  approve.title = 'Approve the boxes on this capture (key b)';
  approve.textContent = 'Approve boxes';
  approve.addEventListener('click', () => approveBoxes(cluster, image, imageState.bbox_status !== 'approved'));
  foot.appendChild(approve);
  const link = document.createElement('a');
  link.href = previewUrl(image, 'edit');
  link.target = '_blank';
  link.rel = 'noreferrer';
  link.textContent = 'open full preview';
  foot.appendChild(link);
  card.addEventListener('pointerdown', () => {
    if (state.focus !== imageIndex && !state.dragging) {
      if (hasDraft()) commitDraft();
      state.focus = imageIndex;
      state.selectedBox = null;
      render();
    }
  });
  if (editable && image.width && image.height) attachBoxEditing(stage, cluster, image);
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
    return;
  }
  state.focus = Math.max(0, Math.min(state.focus, cluster.images.length - 1));
  const decision = decisionFor(cluster.cluster_id);
  const confidence = cluster.minimum_confidence === null
    ? 'no accepted registration'
    : `weakest link ${cluster.minimum_confidence.toFixed(3)}`;
  host.innerHTML = '';
  const panel = document.createElement('section');
  panel.className = 'cluster';
  panel.innerHTML =
    `<h2>${cluster.cluster_id} <span class='status-chip ${decision.status}'>${decision.status}</span></h2>`
    + `<p class='meta'>${cluster.image_count} captures · folder <code>${cluster.source_folder}</code> · ${confidence}`
    + `${cluster.review_reasons.length ? ` · ${cluster.review_reasons.join('; ')}` : ''}`
    + `${state.viewMode === 'focus' ? ` · capture ${state.focus + 1} of ${cluster.image_count}` : ''}</p>`;
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

function attachBoxEditing(stage, cluster, image) {
  const canvas = stage.querySelector('.image-canvas');
  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const draft = openDraft(cluster, image);
    const boxNode = event.target.closest('.box');
    const handle = event.target.classList.contains('handle') ? event.target : null;
    const [startX, startY] = stageToImage(stage, image, event.clientX, event.clientY);
    let mode = 'create';
    let edge = 'se';
    let boxIndex;
    if (handle) {
      mode = 'resize';
      edge = handle.dataset.edge;
      boxIndex = Number(handle.dataset.boxIndex);
    } else if (boxNode) {
      mode = 'move';
      boxIndex = Number(boxNode.dataset.boxIndex);
    } else {
      draft.boxes.push({
        box_id: `box_${draft.boxes.length + 1}`,
        bbox: [startX, startY, startX + MINIMUM_SIDE, startY + MINIMUM_SIDE],
        kind: 'reviewer',
        origin: 'reviewer',
      });
      boxIndex = draft.boxes.length - 1;
    }
    state.selectedBox = boxIndex;
    state.dragging = true;
    // Read the anchor from the draft at drag start, so a drag that follows an
    // earlier edit continues from the current geometry rather than a stale copy.
    const anchor = [...draft.boxes[boxIndex].bbox];
    stage.setPointerCapture?.(event.pointerId);
    let moved = false;

    const onMove = (moveEvent) => {
      const [x, y] = stageToImage(stage, image, moveEvent.clientX, moveEvent.clientY);
      const box = draft.boxes[boxIndex];
      if (!box) return;
      moved = true;
      if (mode === 'create') {
        box.bbox = [Math.min(startX, x), Math.min(startY, y), Math.max(startX, x), Math.max(startY, y)];
      } else if (mode === 'resize') {
        box.bbox = resizeBox(anchor, edge, x, y, image);
      } else {
        const width = anchor[2] - anchor[0];
        const height = anchor[3] - anchor[1];
        const left = Math.max(0, Math.min(image.width - width, anchor[0] + (x - startX)));
        const top = Math.max(0, Math.min(image.height - height, anchor[1] + (y - startY)));
        box.bbox = [left, top, left + width, top + height];
      }
      renderBoxes(canvas, image, draft.boxes, true);
      renderDraftBanner();
    };

    const finish = (cancelled) => {
      stage.removeEventListener('pointermove', onMove);
      stage.removeEventListener('pointerup', onUp);
      stage.removeEventListener('pointercancel', onCancel);
      state.dragging = false;
      if (cancelled) { cancelDraft(); return; }
      const box = draft.boxes[boxIndex];
      if (mode === 'create' && (!moved || box.bbox[2] - box.bbox[0] < MINIMUM_SIDE * 2 || box.bbox[3] - box.bbox[1] < MINIMUM_SIDE * 2)) {
        draft.boxes.splice(boxIndex, 1);
        state.selectedBox = null;
        if (!moved) { render(); return; }
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
  if (resize) {
    box.bbox = resizeBox(box.bbox, 'se', box.bbox[2] + dx * stepSize, box.bbox[3] + dy * stepSize, image);
  } else {
    const width = box.bbox[2] - box.bbox[0];
    const height = box.bbox[3] - box.bbox[1];
    const left = Math.max(0, Math.min(image.width - width, box.bbox[0] + dx * stepSize));
    const top = Math.max(0, Math.min(image.height - height, box.bbox[1] + dy * stepSize));
    box.bbox = [left, top, left + width, top + height];
  }
  render();
  scheduleCommit(700);
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
  el('toggle-focused').addEventListener('click', toggleFocused);
  el('view-mode').addEventListener('click', toggleViewMode);
  el('prev-capture').addEventListener('click', () => flipCapture(-1));
  el('next-capture').addEventListener('click', () => flipCapture(1));
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
      case 'Enter': event.preventDefault(); if (hasDraft()) commitDraft(); break;
      case 'Escape': event.preventDefault(); exitEditing(); break;
      case 'Delete': case 'Backspace': event.preventDefault(); deleteSelectedBox(); break;
      case 'ArrowRight':
        event.preventDefault();
        if (state.editing && state.selectedBox !== null) nudgeSelectedBox(1, 0, event.shiftKey);
        else flipCapture(1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        if (state.editing && state.selectedBox !== null) nudgeSelectedBox(-1, 0, event.shiftKey);
        else flipCapture(-1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        if (state.editing && state.selectedBox !== null) nudgeSelectedBox(0, 1, event.shiftKey);
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (state.editing && state.selectedBox !== null) nudgeSelectedBox(0, -1, event.shiftKey);
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
<header class='bar'>
  <h1>Cluster review</h1>
  <div class='controls'>
    <button type='button' id='prev' title='Previous cluster (key k)'>◀ Prev <kbd>k</kbd></button>
    <button type='button' id='next' title='Next cluster (key j)'>Next <kbd>j</kbd> ▶</button>
    <button type='button' id='approve' title='Mark this grouping correct and advance (key a)'>Mark OK <kbd>a</kbd></button>
    <button type='button' id='dissolve' title='Reject the whole grouping and advance (key d)'>Dissolve <kbd>d</kbd></button>
    <button type='button' id='mark-irregular' class='irregular' title='Mark this grouping irregular and exclude its images from training and validation (key i)'>Irregular <kbd>i</kbd></button>
    <button type='button' id='toggle-focused' title='Include or exclude the focused capture (key x, or its number key)'>Toggle capture <kbd>x</kbd></button>
    <button type='button' id='undo' title='Undo the last saved change (Ctrl+Z)'>Undo <kbd>Ctrl</kbd>+<kbd>Z</kbd></button>
    <button type='button' id='reopen' title='Return this cluster to unreviewed (key r)'>Reopen <kbd>r</kbd></button>
  </div>
  <div class='controls'>
    <button type='button' id='view-mode' aria-pressed='false' title='Switch between side-by-side and full-screen single-capture view (key v)'>View: <span id='view-mode-label'>Side by side</span> <kbd>v</kbd></button>
    <button type='button' id='prev-capture' title='Previous capture in this cluster (Left arrow)'>◀ Capture <kbd>←</kbd></button>
    <button type='button' id='next-capture' title='Next capture in this cluster (Right arrow)'>Capture <kbd>→</kbd> ▶</button>
    <button type='button' id='edit-mode' aria-pressed='false' title='Toggle bbox editing: drag empty space to create, drag a box to move, drag any edge or corner handle to resize (key e)'>Edit boxes <kbd>e</kbd></button>
    <button type='button' id='select-box' title='Select the next box on the focused capture (Tab)'>Select box <kbd>Tab</kbd></button>
    <button type='button' id='approve-boxes' title='Approve the boxes on the focused capture (key b)'>Approve boxes <kbd>b</kbd></button>
    <button type='button' id='delete-box' class='danger' title='Delete the selected box (Delete)'>Delete box <kbd>Del</kbd></button>
  </div>
  <div class='controls'>
    <label class='field'>Search<input id='search' type='search' placeholder='cluster, folder, reason'></label>
    <label class='field'>Minimum size<select id='min-size'>
      <option value='1'>all clusters</option><option value='2'>2+ images</option>
      <option value='3'>3+ images</option><option value='4'>4+ images</option>
      <option value='10'>10+ images</option></select></label>
    <label class='field'>Sort<select id='sort'>
      <option value='confidence-asc'>weakest confidence first</option>
      <option value='confidence-desc'>strongest confidence first</option>
      <option value='size-desc'>largest first</option><option value='size-asc'>smallest first</option>
      <option value='folder'>source folder</option><option value='id'>cluster ID</option></select></label>
    <label class='field'>Status<select id='status-filter'>
      <option value=''>all</option><option value='unreviewed'>unreviewed</option>
      <option value='reviewed'>reviewed</option><option value='approved'>approved</option>
      <option value='edited'>edited</option><option value='dissolved'>dissolved</option>
      <option value='irregular'>irregular</option></select></label>
  </div>
  <span id='save-state' class='save-state ok' role='status' aria-live='polite'>Loading</span>
</header>
<main id='main'>
  <div>
    <div id='draft-banner' class='draft-banner' role='status' aria-live='polite' hidden>
      <span id='draft-summary'></span>
      <button type='button' id='draft-save' class='primary' title='Save this box edit now (Enter)'>Save boxes <kbd>Enter</kbd></button>
      <button type='button' id='draft-cancel' title='Discard this box edit (Escape)'>Cancel <kbd>Esc</kbd></button>
      <span>Releasing the mouse saves automatically.</span>
    </div>
    <p class='progress'><span id='queue-position'></span> · <span id='progress'></span></p>
    <div id='cluster-host'></div>
  </div>
  <aside>
    <h3>Keyboard</h3>
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
      <tr><td><kbd>Ctrl</kbd>+<kbd>Z</kbd></td><td>undo last change</td></tr>
      <tr><td><kbd>e</kbd></td><td>bbox edit mode</td></tr>
      <tr><td><kbd>Tab</kbd></td><td>select next box</td></tr>
      <tr><td><kbd>b</kbd></td><td>approve boxes on capture</td></tr>
      <tr><td><kbd>Del</kbd></td><td>delete selected box</td></tr>
      <tr><td>arrows</td><td>nudge selected box in edit mode</td></tr>
      <tr><td><kbd>Shift</kbd>+arrows</td><td>resize selected box</td></tr>
      <tr><td><kbd>Enter</kbd></td><td>save a pending box edit</td></tr>
      <tr><td><kbd>Esc</kbd></td><td>cancel edit / clear selection</td></tr>
    </table>
    <p class='hint'>Select a box to get eight drag handles: four corners and four edges. Dropping a 2-image cluster to one capture dissolves it automatically. Editing membership marks the cluster reviewed; box approval is tracked separately.</p>
    <h3>Finish the pass</h3>
    <div class='controls'>
      <button type='button' id='mark-remaining' title='Approve every cluster still unreviewed'>Mark remaining clusters OK</button>
      <button type='button' id='mark-remaining-boxes' title='Approve every box set still unreviewed'>Mark remaining boxes OK</button>
      <button type='button' id='export' title='Write corrected cluster and box manifests'>Export corrected manifests</button>
    </div>
    <p class='hint'>Autosaves to <code id='decisions-path'></code></p>
  </aside>
</main>
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
