import {COMPOSITION_SCHEMA, buildCompositionConfig, buildQuestion, preferredModelId, simpleCriteriaText, validateCompositionConfig} from "./logic.mjs?v=20261003-composer8";
import {attachNode, branchLabels, canMoveNode, createNode, executeWorkflow, layoutTree, moveNode, removeNode, replaceNode, walkTree} from "./composition.mjs?v=20261003-composer8";

const TYPES = {choice: "Choice", score: "Score", noul: "Noul"};
const STORAGE_KEY = "jevembed.composition.v2";
const SVG_NS = "http://www.w3.org/2000/svg";
const ICONS = {
  plus: ["M12 5v14", "M5 12h14"],
  minus: ["M5 12h14"],
  undo: ["M9 7 4 12l5 5", "M5 12h8a6 6 0 0 1 6 6"],
  refresh: ["M20 11a8 8 0 0 0-14.7-4L4 9", "M4 4v5h5", "M4 13a8 8 0 0 0 14.7 4L20 15", "M20 20v-5h-5"],
  check: ["m5 12 4 4L19 6"],
  move: ["M5 9h14", "m15 5 4 4-4 4", "M19 15H5", "m9 4-4-4 4-4"],
  schema: ["M5 4h14v16H5z", "M8 8h8", "M8 12h8", "M8 16h5"],
  trash: ["M3 6h18", "M9 6V3h6v3", "M5 6l1 15h12l1-15", "M10 10v7", "M14 10v7"],
  fit: ["M8 3H5a2 2 0 0 0-2 2v3", "M16 3h3a2 2 0 0 1 2 2v3", "M3 16v3a2 2 0 0 0 2 2h3", "M21 16v3a2 2 0 0 1-2 2h-3"],
  zoomIn: ["M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Z", "M11 7v8", "M7 11h8", "m16 16 4 4"],
  zoomOut: ["M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Z", "M7 11h8", "m16 16 4 4"],
  download: ["M12 3v12", "m7 10 5 5 5-5", "M5 20h14"],
  upload: ["M12 16V4", "m7 9 5-5 5 5", "M5 20h14"],
  copy: ["M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3", "M5 8h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2Z"],
  play: ["m8 5 11 7-11 7z"],
  stop: ["M7 7h10v10H7z"]
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.7");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  for (const pathData of ICONS[name] || ICONS.plus) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", pathData);
    svg.append(path);
  }
  return svg;
}

function scoreDescription(node, label) {
  if (typeof node.criteria === "string")
    return node.criteria.split(/\r?\n/)[Number(label)]?.trim() || "";
  return String(node.criteria?.[Number(label)] ?? "");
}

function previewInstructions(node) {
  if (!node.instructionsAdvanced) return node.instructions;
  try {
    const parsed = JSON.parse(node.instructions);
    return typeof parsed === "string" ? parsed : JSON.stringify(parsed);
  } catch { return node.instructions; }
}

function exampleRoot() {
  return {
    id: "route", type: "choice",
    instructions: "Which team should handle this request?",
    criteria: "support: Support needs\nproduct: Product defects\naccount: Account help",
    advanced: false,
    children: {
      support: [{
        id: "support_severity", type: "score",
        instructions: "How urgent is the support request?",
        criteria: "Routine\nTime-sensitive\nBlocking", advanced: false,
        children: {"2": [{
          id: "support_escalation", type: "noul",
          instructions: "Should a person step in?", criteria: "null", advanced: true,
          children: {true: [{
            id: "human_handoff", type: "choice",
            instructions: "Where should the request go?",
            criteria: "agent: Support agent\nspecialist: Product specialist",
            advanced: false
          }]}
        }]}
      }],
      product: [{
        id: "product_impact", type: "choice",
        instructions: "What is affected?",
        criteria: "export: Export flow\nsettings: Settings", advanced: false
      }],
      account: [{
        id: "account_review", type: "noul",
        instructions: "Does the customer need identity verification?",
        criteria: "null", advanced: true
      }]
    }
  };
}

function draftFromCanonical(node) {
  let criteria = node.type === "noul" ? "null" : "";
  let advanced = node.type === "noul";
  if (Object.hasOwn(node, "criteria")) {
    try { criteria = simpleCriteriaText(node); }
    catch {
      criteria = JSON.stringify(node.criteria, null, 2);
      advanced = true;
    }
  }
  const structuredInstructions = typeof node.instructions !== "string";
  const result = {id: node.id, type: node.type,
    instructions: structuredInstructions ? JSON.stringify(node.instructions, null, 2) : node.instructions,
    instructionsAdvanced: structuredInstructions, criteria, advanced, children: {}};
  for (const [branch, children] of Object.entries(node.children || {}))
    result.children[branch] = children.map(draftFromCanonical);
  return result;
}

function canonicalNode(node) {
  const question = buildQuestion(node);
  const result = {id: node.id, ...question};
  const children = Object.fromEntries(Object.entries(node.children || {})
    .filter(([, items]) => items.length)
    .map(([branch, items]) => [branch, items.map(canonicalNode)]));
  if (Object.keys(children).length) result.children = children;
  return result;
}

function loadDraft() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (!saved?.root) return null;
    const config = validateCompositionConfig({
      version: 1, model: saved.model || "preview", state: saved.state, root: saved.root
    });
    return {
      root: draftFromCanonical(config.root),
      model: saved.model || "",
      state: config.state,
      jsonState: Boolean(saved.jsonState) || typeof config.state !== "string"
    };
  } catch { return null; }
}

export function initComposer(container) {
  const initial = loadDraft();
  let root = initial?.root || exampleRoot();
  let model = initial?.model || "";
  let models = [];
  let modelError = "";
  let selectedId = root.id;
  let mode = "inspector";
  let zoom = 0.82;
  let overviewZoom = 1;
  let controller = null;
  let running = false;
  let pointerDrag = null;
  let dragFrame = null;
  let layout = null;
  let overviewLayout = null;
  let jsonDirty = false;
  let staleResults = false;
  let resultRows = [];
  let resultSkipped = [];
  let runStatus = "";
  let undoStack = [];

  function restoreLegacyText(value) {
    try {
      const decoded = JSON.parse(value);
      if (typeof decoded === "string") return decoded;
    } catch {}
    return value;
  }

  let stateText = typeof initial?.state === "string"
    ? (initial.jsonState ? JSON.stringify(initial.state, null, 2) : restoreLegacyText(initial.state))
    : initial?.state !== undefined
      ? JSON.stringify(initial.state, null, 2)
      : "A customer reports an urgent export issue and asks for a specialist.";
  let jsonState = initial?.jsonState || false;

  container.innerHTML = "<header class='composer-heading'><div><p class='composer-kicker'>JEVEMBED / WORKFLOW</p><h1>Decision composer</h1></div><button class='composer-button composer-example' data-action='example' type='button'>Load example</button></header><section class='composer-context'><div class='composer-context-title'><span>01</span><div><h2>Shared context</h2><p>All decisions evaluate this input.</p></div></div><div class='composer-context-fields'><label class='composer-field'>Model<select data-model aria-label='Embedding model'></select></label><label class='composer-check'><input type='checkbox' data-json-toggle><span>JSON context</span></label><label class='composer-field composer-state-field'>State<textarea data-state rows='3' spellcheck='true'></textarea></label></div></section><main class='composer-layout'><section class='composer-board' aria-label='Decision map'><header class='composer-panel-heading'><div><span class='composer-step-number'>02</span><div><h2>Decision map</h2><p data-count></p></div></div><div class='composer-board-actions'><button class='composer-button composer-add-button' data-action='add' type='button'><span aria-hidden='true'>+</span> Add decision</button><button class='composer-icon-button' data-action='undo' type='button' title='Undo change' aria-label='Undo change' disabled></button><button class='composer-icon-button' data-action='download' type='button' title='Download workflow JSON' aria-label='Download workflow JSON'></button><button class='composer-icon-button' data-action='import' type='button' title='Import workflow JSON' aria-label='Import workflow JSON'></button><input data-file type='file' accept='.json,application/json' hidden></header><div class='composer-map-toolbar'><span>Drop a decision on a result to move it.</span><div data-zoom></div></div><div class='composer-map-scroll' data-map-scroll tabindex='0' role='region' aria-label='Decision map'><div class='composer-map-extent' data-map-extent><div class='composer-map-world' data-map-stage><svg class='composer-map-lines' data-lines aria-hidden='true'></svg><div class='composer-map-nodes' data-nodes></div></div></div></div><footer class='composer-map-footer'>Only the selected result branch continues to its child decisions.</footer></section><aside class='composer-side-panel'><header class='composer-panel-heading composer-side-heading'><div><span class='composer-step-number'>03</span><div><h2>Workflow details</h2><p data-inspector-heading></p></div></div><button class='composer-icon-button' data-action='schema' type='button' title='Download JSON Schema' aria-label='Download JSON Schema'></button></header><div class='composer-tabs' role='tablist' aria-label='Workflow details'><button type='button' role='tab' data-tab='inspector'>Inspector</button><button type='button' role='tab' data-tab='json'>JSON</button><button type='button' role='tab' data-tab='overview'>Global preview</button></div><section class='composer-inspector' data-inspector></section><section class='composer-json-panel' data-json-panel hidden><div class='composer-json-title'><strong>Workflow configuration</strong><button class='composer-icon-button' data-action='copy' type='button' title='Copy JSON' aria-label='Copy JSON'></button></div><textarea class='composer-json-editor' data-json-editor spellcheck='false' aria-label='Workflow JSON'></textarea><p class='composer-validation' data-json-error hidden></p><button class='composer-button composer-apply-json' data-action='apply-json' type='button'>Apply JSON</button></section><section class='composer-overview-panel' data-overview-panel hidden><div class='composer-overview-title'><span>Complete workflow</span><button class='composer-icon-button' data-action='fit-overview' type='button' title='Fit workflow' aria-label='Fit workflow'></button></div><div class='composer-map-scroll composer-overview-scroll' data-overview-scroll tabindex='0' role='region' aria-label='Global workflow preview'><div class='composer-map-extent' data-overview-extent><div class='composer-map-world' data-overview-stage><svg class='composer-map-lines' data-overview-lines aria-hidden='true'></svg><div class='composer-map-nodes' data-overview-nodes></div></div></div></div></section><section class='composer-run-panel'><div class='composer-run-controls'><button class='composer-run-button' data-action='run' type='button'></button><button class='composer-icon-button' data-action='stop' type='button' title='Stop workflow' aria-label='Stop workflow' hidden></button></div><p class='composer-run-status' data-status role='status' aria-live='polite'></p><section class='composer-results' data-results hidden></section></section></aside></main><input data-import-text type='file' accept='.json,application/json' hidden><p class='composer-live' data-live role='status' aria-live='polite'></p>";

  const $ = selector => container.querySelector(selector);
  const ui = {
    model: $("[data-model]"), jsonToggle: $("[data-json-toggle]"), state: $("[data-state]"),
    count: $("[data-count]"), mapScroll: $("[data-map-scroll]"), mapExtent: $("[data-map-extent]"),
    mapStage: $("[data-map-stage]"), mapLines: $("[data-lines]"), nodes: $("[data-nodes]"),
    inspector: $("[data-inspector]"), inspectorHeading: $("[data-inspector-heading]"),
    tabs: [...container.querySelectorAll("[data-tab]")],
    jsonPanel: $("[data-json-panel]"), jsonEditor: $("[data-json-editor]"), jsonError: $("[data-json-error]"),
    overviewPanel: $("[data-overview-panel]"), overviewScroll: $("[data-overview-scroll]"),
    overviewExtent: $("[data-overview-extent]"), overviewStage: $("[data-overview-stage]"),
    overviewLines: $("[data-overview-lines]"), overviewNodes: $("[data-overview-nodes]"),
    zoom: $("[data-zoom]"), undo: $("[data-action=undo]"), run: $("[data-action=run]"),
    stop: $("[data-action=stop]"), status: $("[data-status]"), results: $("[data-results]"),
    live: $("[data-live]")
  };

  function addButtonIcon(button, name) {
    if (!button || button.querySelector("svg")) return;
    button.prepend(icon(name));
    button.classList.add("has-icon");
  }

  addButtonIcon($("[data-action=example]"), "refresh");
  addButtonIcon(ui.undo, "undo");
  addButtonIcon($("[data-action=download]"), "download");
  addButtonIcon($("[data-action=import]"), "upload");
  addButtonIcon($("[data-action=schema]"), "schema");
  addButtonIcon($("[data-action=copy]"), "copy");
  addButtonIcon($("[data-action=apply-json]"), "check");
  addButtonIcon($("[data-action=fit-overview]"), "fit");
  addButtonIcon(ui.stop, "stop");
  $("[data-action=add]").querySelector("span")?.remove();
  addButtonIcon($("[data-action=add]"), "plus");
  $(".composer-map-toolbar > span").textContent = "Drag MOVE onto a result or above / below another decision.";

  function currentConfig(useFallbackModel = false) {
    return buildCompositionConfig({
      model: model || (useFallbackModel ? "preview" : ""),
      state: ui.state.value,
      jsonState: ui.jsonToggle.checked,
      root
    });
  }

  function announce(message, isError = false) {
    ui.live.textContent = message;
    ui.live.classList.toggle("is-error", isError);
  }

  function setStatus(message, isError = false) {
    runStatus = message;
    ui.status.textContent = message;
    ui.status.classList.toggle("is-error", isError);
  }

  function remember() {
    undoStack.push(JSON.parse(JSON.stringify(root)));
    if (undoStack.length > 50) undoStack.shift();
    ui.undo.disabled = !undoStack.length;
  }

  function persist() {
    try {
      const config = currentConfig(true);
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        ...config, jsonState: ui.jsonToggle.checked
      }));
    } catch {}
  }

  function updateModels() {
    const preferred = model || preferredModelId(models);
    ui.model.replaceChildren();
    for (const entry of models) ui.model.add(new Option(entry.id, entry.id));
    if (preferred && !models.some(entry => entry.id === preferred)) {
      const option = new Option(preferred + " (unavailable)", preferred);
      ui.model.add(option);
    }
    if (!preferred && !models.length)
      ui.model.add(new Option(modelError || "No models available", ""));
    model = preferred;
    ui.model.value = preferred;
    renderRun();
  }

  function moveDecision(nodeId, parentId, branch, index) {
    const previous = structuredClone(root);
    try {
      moveNode(root, nodeId, parentId, branch, index);
      rememberSnapshot(previous);
      selectedId = nodeId;
      staleResults = resultRows.length > 0;
      jsonDirty = false;
      persist();
      render();
      announce("Decision moved to " + parentId + " / " + branch + ".");
    } catch (error) {
      announce(error.message, true);
    }
  }

  function clearDropTargets() {
    ui.nodes.querySelectorAll(".is-drop-target, .is-insert-before, .is-insert-after")
      .forEach(target => target.classList.remove("is-drop-target", "is-insert-before", "is-insert-after"));
  }

  function findDropTarget(clientX, clientY) {
    const hit = document.elementFromPoint(clientX, clientY);
    if (!hit || !ui.nodes.contains(hit)) return null;
    const branchSlot = hit.closest(".composer-branch-slot");
    if (branchSlot) {
      const {parent: parentId, branch} = branchSlot.dataset;
      return canMoveNode(root, pointerDrag.id, parentId, branch)
        ? {parentId, branch, element: branchSlot, className: "is-drop-target"} : null;
    }
    const card = hit.closest(".composer-node");
    const target = layout.nodes.find(item => item.node.id === card?.dataset.node);
    if (!target?.parentId || target.node.id === pointerDrag.id ||
        !canMoveNode(root, pointerDrag.id, target.parentId, target.branch)) return null;
    const parent = walkTree(root).find(entry => entry.node.id === target.parentId).node;
    const siblings = parent.children[target.branch].filter(node => node.id !== pointerDrag.id);
    const before = clientY < card.getBoundingClientRect().top + card.getBoundingClientRect().height / 2;
    return {parentId: target.parentId, branch: target.branch,
      index: siblings.findIndex(node => node.id === target.node.id) + (before ? 0 : 1),
      element: card, className: before ? "is-insert-before" : "is-insert-after"};
  }

  function updatePointerDrag() {
    if (!pointerDrag?.active) return;
    pointerDrag.ghost.style.transform = "translate(" +
      (pointerDrag.clientX - pointerDrag.offsetX) + "px, " +
      (pointerDrag.clientY - pointerDrag.offsetY) + "px) scale(" + zoom + ")";
    clearDropTargets();
    pointerDrag.target = findDropTarget(pointerDrag.clientX, pointerDrag.clientY);
    pointerDrag.target?.element.classList.add(pointerDrag.target.className);
  }

  function scrollDuringDrag() {
    dragFrame = null;
    if (!pointerDrag?.active) return;
    const bounds = ui.mapScroll.getBoundingClientRect();
    const inside = pointerDrag.clientX >= bounds.left - 24 && pointerDrag.clientX <= bounds.right + 24 &&
      pointerDrag.clientY >= bounds.top - 24 && pointerDrag.clientY <= bounds.bottom + 24;
    if (inside) {
      const speed = (position, start, end) => position < start + 36
        ? -Math.min(16, (start + 36 - position) / 3)
        : position > end - 36 ? Math.min(16, (position - end + 36) / 3) : 0;
      ui.mapScroll.scrollLeft += speed(pointerDrag.clientX, bounds.left, bounds.right);
      ui.mapScroll.scrollTop += speed(pointerDrag.clientY, bounds.top, bounds.bottom);
    }
    updatePointerDrag();
    dragFrame = requestAnimationFrame(scrollDuringDrag);
  }

  function endPointerDrag(commit = false) {
    if (!pointerDrag) return;
    const drag = pointerDrag;
    const target = commit && drag.active ? findDropTarget(drag.clientX, drag.clientY) : null;
    pointerDrag = null;
    if (dragFrame !== null) cancelAnimationFrame(dragFrame);
    dragFrame = null;
    drag.ghost?.remove();
    drag.card.classList.remove("is-dragging");
    container.classList.remove("is-pointer-dragging");
    clearDropTargets();
    if (drag.handle.hasPointerCapture(drag.pointerId)) drag.handle.releasePointerCapture(drag.pointerId);
    if (target) moveDecision(drag.id, target.parentId, target.branch, target.index);
    else if (commit && drag.active) announce("Drop on a result branch or above / below another decision.");
  }

  ui.nodes.addEventListener("pointerdown", event => {
    if (event.button !== 0 || pointerDrag) return;
    const handle = event.target.closest(".composer-grip");
    const card = handle?.closest(".composer-node");
    const item = layout?.nodes.find(entry => entry.node.id === card?.dataset.node);
    if (!item?.parentId) return;
    event.preventDefault();
    const bounds = card.getBoundingClientRect();
    pointerDrag = {id: item.node.id, pointerId: event.pointerId, handle, card, active: false,
      startX: event.clientX, startY: event.clientY, clientX: event.clientX, clientY: event.clientY,
      offsetX: event.clientX - bounds.left, offsetY: event.clientY - bounds.top};
    handle.setPointerCapture(event.pointerId);
  });

  ui.nodes.addEventListener("pointermove", event => {
    if (!pointerDrag || event.pointerId !== pointerDrag.pointerId) return;
    pointerDrag.clientX = event.clientX;
    pointerDrag.clientY = event.clientY;
    if (!pointerDrag.active && Math.hypot(event.clientX - pointerDrag.startX, event.clientY - pointerDrag.startY) >= 5) {
      pointerDrag.active = true;
      const ghost = pointerDrag.card.cloneNode(true);
      ghost.classList.remove("is-selected");
      ghost.classList.add("composer-drag-ghost");
      ghost.removeAttribute("data-node");
      ghost.setAttribute("aria-hidden", "true");
      ghost.querySelectorAll("button").forEach(button => button.tabIndex = -1);
      ghost.style.left = "0";
      ghost.style.top = "0";
      container.append(ghost);
      pointerDrag.ghost = ghost;
      pointerDrag.card.classList.add("is-dragging");
      container.classList.add("is-pointer-dragging");
      dragFrame = requestAnimationFrame(scrollDuringDrag);
    }
    if (pointerDrag.active) {
      event.preventDefault();
      updatePointerDrag();
    }
  });
  ui.nodes.addEventListener("pointerup", event => {
    if (event.pointerId !== pointerDrag?.pointerId) return;
    pointerDrag.clientX = event.clientX;
    pointerDrag.clientY = event.clientY;
    endPointerDrag(true);
  });
  ui.nodes.addEventListener("pointercancel", event => {
    if (event.pointerId === pointerDrag?.pointerId) endPointerDrag();
  });
  ui.nodes.addEventListener("lostpointercapture", event => {
    if (event.pointerId === pointerDrag?.pointerId) endPointerDrag();
  });
  container.addEventListener("keydown", event => {
    if (event.key === "Escape" && pointerDrag) {
      event.preventDefault();
      endPointerDrag();
    }
  });

  function drawMap(stage, extent, lines, nodeParent, tree, scale, interactive) {
    if (!tree) return;
    const oldScroll = interactive ? [ui.mapScroll.scrollLeft, ui.mapScroll.scrollTop] : null;
    extent.style.width = Math.ceil(tree.width * scale) + "px";
    extent.style.height = Math.ceil(tree.height * scale) + "px";
    stage.style.width = tree.width + "px";
    stage.style.height = tree.height + "px";
    stage.style.transform = "scale(" + scale + ")";
    nodeParent.replaceChildren();
    lines.replaceChildren();
    lines.setAttribute("viewBox", "0 0 " + tree.width + " " + tree.height);
    lines.setAttribute("width", String(tree.width));
    lines.setAttribute("height", String(tree.height));

    for (const edge of tree.edges) {
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", "M " + edge.fromX + " " + edge.fromY + " C " +
        (edge.fromX + 48) + " " + edge.fromY + ", " +
        (edge.toX - 48) + " " + edge.toY + ", " + edge.toX + " " + edge.toY);
      path.setAttribute("class", "composer-link");
      lines.append(path);
      if (!interactive) continue;
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(edge.fromX + 12));
      label.setAttribute("y", String(edge.fromY - 5));
      label.setAttribute("class", "composer-link-label");
      const parent = tree.nodes.find(item => item.node.id === edge.parentId)?.node;
      label.textContent = parent?.type === "score"
        ? edge.branch + " · " + scoreDescription(parent, edge.branch) : edge.branch;
      lines.append(label);
    }

    for (const item of tree.nodes) {
      if (!interactive) {
        const point = element("button", "composer-overview-point composer-node-" + item.node.type +
          (item.node.id === selectedId ? " is-selected" : ""));
        point.type = "button";
        point.dataset.overviewNode = item.node.id;
        point.style.left = item.x + "px";
        point.style.top = item.y + "px";
        point.title = TYPES[item.node.type] + " / " + item.node.id;
        point.setAttribute("aria-label", "Inspect " + TYPES[item.node.type] + " decision " + item.node.id);
        point.append(element("span", "composer-overview-label", item.node.id));
        point.addEventListener("click", () => {
          selectedId = item.node.id;
          mode = "inspector";
          render();
          requestAnimationFrame(() => ui.nodes.querySelector('[data-node="' + item.node.id + '"]')
            ?.scrollIntoView({block: "nearest", inline: "center"}));
        });
        nodeParent.append(point);
        continue;
      }
      const card = element("article", "composer-node composer-node-" + item.node.type +
        (item.node.id === selectedId ? " is-selected" : "") +
        (item.depth === 0 ? " is-root" : ""));
      card.dataset.node = item.node.id;
      card.style.left = item.x + "px";
      card.style.top = item.y + "px";
      card.style.width = item.width + "px";
      card.style.minHeight = item.height + "px";
      card.draggable = false;
      const heading = element("div", "composer-node-heading");
      const grip = element("span", "composer-grip", item.depth ? "MOVE" : "ROOT");
      grip.setAttribute("aria-label", item.depth ? "Drag to move decision" : "Root decision");
      const title = element("button", "composer-node-title", TYPES[item.node.type] + " · " + item.node.id);
      title.type = "button";
      title.addEventListener("click", event => {
        event.stopPropagation();
        selectedId = item.node.id;
        mode = "inspector";
        render();
      });
      heading.append(grip, title);
      card.append(heading, element("p", "composer-node-question", previewInstructions(item.node)));

      for (const label of item.labels) {
        const children = item.node.children?.[label] || [];
        const branch = element("div", "composer-branch-slot" + (children.length ? " has-children" : ""));
        branch.dataset.parent = item.node.id;
        branch.dataset.branch = label;
        branch.append(element("span", "composer-branch-marker"),
          element("span", "composer-branch-label", item.node.type === "score"
            ? label + " · " + scoreDescription(item.node, label) : label));
        branch.append(element("span", "composer-branch-count", children.length ? String(children.length) : ""));
        if (interactive) {
          const add = element("button", "composer-branch-add");
          add.type = "button";
          add.append(icon("plus"));
          add.title = "Add decision to " + label;
          add.setAttribute("aria-label", "Add decision to " + label);
          add.addEventListener("click", event => {
            event.stopPropagation();
            try {
              const child = createNode(root);
              remember();
              attachNode(root, item.node.id, label, child);
              selectedId = child.id;
              persist();
              render();
            } catch (error) { announce(error.message, true); }
          });
          branch.append(add);
        }
        card.append(branch);
      }
      nodeParent.append(card);
    }
    if (oldScroll) {
      ui.mapScroll.scrollLeft = oldScroll[0];
      ui.mapScroll.scrollTop = oldScroll[1];
    }
  }

  function drawMaps() {
    if (!layout) return;
    drawMap(ui.mapStage, ui.mapExtent, ui.mapLines, ui.nodes, layout, zoom, true);
    if (mode === "overview")
      drawMap(ui.overviewStage, ui.overviewExtent, ui.overviewLines, ui.overviewNodes, overviewLayout, overviewZoom, false);
  }

  function renderCount() {
    const entries = walkTree(root);
    const depth = entries.reduce((max, entry) => Math.max(max, entry.depth), 0) + 1;
    ui.count.textContent = entries.length + " decisions · " + depth + (depth === 1 ? " level" : " levels");
  }

  function renderInspector() {
    const entry = walkTree(root).find(item => item.node.id === selectedId);
    if (!entry) selectedId = root.id;
    const selected = walkTree(root).find(item => item.node.id === selectedId);
    const node = selected.node;
    ui.inspectorHeading.textContent = selected.parent
      ? "Level " + (selected.depth + 1) + " · " + selected.branch
      : "Root decision";
    ui.inspector.replaceChildren();
    const fields = element("div", "composer-fields");
    const idLabel = element("label", "composer-field", "Decision ID");
    const id = element("input", "composer-input");
    id.value = node.id;
    id.autocomplete = "off";
    idLabel.append(id);
    const typeLabel = element("label", "composer-field", "Decision type");
    const type = element("select", "composer-input");
    for (const value of ["choice", "score", "noul"]) type.add(new Option(TYPES[value], value));
    type.value = node.type;
    typeLabel.append(type);
    const questionLabel = element("label", "composer-field", "Question");
    const question = element("textarea", "composer-input");
    question.rows = 3;
    question.value = node.instructions;
    questionLabel.append(question);
    const instructionsModeLabel = element("label", "composer-check composer-instructions-mode");
    const instructionsMode = element("input");
    instructionsMode.type = "checkbox";
    instructionsMode.checked = Boolean(node.instructionsAdvanced);
    instructionsModeLabel.append(instructionsMode, element("span", "", "JSON instructions"));
    const criteriaLabel = element("label", "composer-field composer-criteria-field",
      node.type === "choice" ? "Choices" : node.type === "score" ? "Score levels" : "Criteria (JSON)");
    const criteria = element("textarea", "composer-input composer-criteria");
    criteria.rows = 5;
    criteria.spellcheck = false;
    criteria.value = node.type === "noul" ? node.criteria?.trim() || "null" : node.criteria || "";
    criteria.placeholder = node.type === "choice" ? "yes: Proceed\nno: Stop" :
      node.type === "score" ? "Low impact\nHigh impact" : "{\"true\":\"Escalate\",\"false\":\"Continue\"}";
    criteriaLabel.append(criteria);
    const jsonModeLabel = element("label", "composer-check composer-criteria-mode");
    const jsonMode = element("input");
    jsonMode.type = "checkbox";
    jsonMode.checked = node.type === "noul" || Boolean(node.advanced);
    jsonModeLabel.hidden = node.type === "noul";
    jsonModeLabel.append(jsonMode, element("span", "", "JSON criteria"));
    const criteriaHint = element("p", "composer-criteria-hint",
      'Use null to omit criteria, or provide both "true" and "false" descriptions.');
    criteriaHint.hidden = node.type !== "noul";
    fields.append(idLabel, typeLabel, questionLabel, instructionsModeLabel, criteriaLabel, jsonModeLabel, criteriaHint);

    const message = element("p", "composer-validation");
    message.hidden = true;
    const apply = element("button", "composer-button composer-apply", "Apply decision");
    apply.type = "button";
    addButtonIcon(apply, "check");
    apply.addEventListener("click", () => {
      const replacement = {
        id: id.value.trim(), type: type.value, instructions: question.value,
        instructionsAdvanced: instructionsMode.checked,
        criteria: criteria.value, advanced: type.value === "noul" || jsonMode.checked,
        children: node.children || {}
      };
      const previous = JSON.parse(JSON.stringify(root));
      try {
        replaceNode(root, node.id, replacement);
        buildCompositionConfig({
          model: model || "preview", state: ui.state.value,
          jsonState: ui.jsonToggle.checked, root
        });
        rememberSnapshot(previous);
        selectedId = replacement.id;
        staleResults = resultRows.length > 0;
        jsonDirty = false;
        persist();
        render();
        announce("Decision updated.");
      } catch (error) {
        root = previous;
        selectedId = node.id;
        message.hidden = false;
        message.textContent = error.message;
      }
    });

    type.addEventListener("change", () => {
      const next = type.value;
      criteriaLabel.firstChild.textContent = next === "choice" ? "Choices" : next === "score" ? "Score levels" : "Criteria (JSON)";
      jsonMode.checked = next === "noul";
      jsonModeLabel.hidden = next === "noul";
      criteriaHint.hidden = next !== "noul";
      message.hidden = true;
      if (next === "choice") criteria.value = "yes: Proceed\nno: Stop";
      else if (next === "score") criteria.value = "Low impact\nHigh impact";
      else criteria.value = "null";
    });
    instructionsMode.addEventListener("change", () => {
      try {
        if (instructionsMode.checked) question.value = JSON.stringify(question.value, null, 2);
        else {
          const parsed = JSON.parse(question.value);
          if (typeof parsed !== "string") throw new Error("Structured instructions must remain in JSON mode.");
          question.value = parsed;
        }
      } catch (error) {
        instructionsMode.checked = !instructionsMode.checked;
        message.hidden = false;
        message.textContent = error.message;
      }
    });
    jsonMode.addEventListener("change", () => {
      try {
        if (jsonMode.checked) {
          const questionValue = buildQuestion({type: type.value, instructions: question.value,
            criteria: criteria.value, advanced: false});
          let raw = questionValue.criteria;
          if (type.value === "noul" && raw === undefined) raw = {true: "Yes", false: "No"};
          criteria.value = JSON.stringify(raw, null, 2);
        } else {
          const raw = JSON.parse(criteria.value);
          criteria.value = simpleCriteriaText({type: type.value, criteria: raw});
        }
      } catch (error) {
        jsonMode.checked = !jsonMode.checked;
        message.hidden = false;
        message.textContent = error.message;
      }
    });

    const actions = element("div", "composer-inspector-actions");
    actions.append(apply);
    const remove = element("button", "composer-button composer-remove", "Remove");
    remove.type = "button";
    remove.disabled = !selected.parent;
    remove.textContent += " " + (walkTree(node).length - 1 || "");
    addButtonIcon(remove, "trash");
    remove.addEventListener("click", () => {
      if (!selected.parent) return;
      remember();
      selectedId = removeNode(root, node.id);
      staleResults = resultRows.length > 0;
      persist();
      render();
    });
    actions.append(remove);
    fields.append(message);

    if (selected.parent) {
      const moveField = element("label", "composer-field composer-move-field", "Move to result");
      const target = element("select", "composer-input");
      target.add(new Option("Choose a result branch", ""));
      const descendants = new Set(walkTree(node).map(item => item.node.id));
      for (const parentEntry of walkTree(root)) {
        if (descendants.has(parentEntry.node.id)) continue;
        for (const branch of branchLabels(parentEntry.node)) {
          const option = new Option(parentEntry.node.id + " · " + branch, parentEntry.node.id + "::" + branch);
          target.add(option);
        }
      }
      const move = element("button", "composer-button composer-move-button", "Move decision");
      move.type = "button";
      addButtonIcon(move, "move");
      move.addEventListener("click", () => {
        const split = target.value.indexOf("::");
        if (split < 0) {
          message.hidden = false;
          message.textContent = "Choose a result branch.";
          return;
        }
        const parentId = target.value.slice(0, split);
        const branch = target.value.slice(split + 2);
        try {
          remember();
          moveNode(root, node.id, parentId, branch);
          staleResults = resultRows.length > 0;
          persist();
          render();
          announce("Decision moved.");
        } catch (error) {
          message.hidden = false;
          message.textContent = error.message;
        }
      });
      moveField.append(target);
      fields.append(moveField, move);
    }
    ui.inspector.append(fields, actions);
  }

  function rememberSnapshot(snapshot) {
    undoStack.push(snapshot);
    if (undoStack.length > 50) undoStack.shift();
    ui.undo.disabled = false;
  }

  function renderJson() {
    let config;
    let error = "";
    try { config = currentConfig(true); }
    catch (exception) { error = exception.message; }
    ui.jsonError.hidden = !error;
    ui.jsonError.textContent = error;
    if (!jsonDirty) {
      ui.jsonEditor.value = config ? JSON.stringify(config, null, 2) : "";
      ui.jsonEditor.setAttribute("aria-invalid", error ? "true" : "false");
    }
  }

  function renderTabs() {
    container.classList.toggle("is-overview", mode === "overview");
    for (const tab of ui.tabs) {
      const active = tab.dataset.tab === mode;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    ui.inspector.hidden = mode !== "inspector";
    ui.jsonPanel.hidden = mode !== "json";
    ui.overviewPanel.hidden = mode !== "overview";
  }

  function renderResults() {
    ui.results.hidden = resultRows.length === 0 && !running;
    ui.results.replaceChildren();
    if (!resultRows.length && !running) return;
    const heading = element("h3", "composer-results-heading", staleResults ? "Last execution path" : "Execution path");
    ui.results.append(heading);
    if (staleResults) ui.results.append(element("p", "composer-stale-note", "The workflow has changed since this run."));
    for (const rowData of resultRows) {
      const row = element("div", "composer-result-row");
      row.append(element("span", "composer-result-type", TYPES[rowData.type]));
      row.append(element("code", "composer-result-id", rowData.id));
      row.append(element("span", "composer-result-prediction", rowData.prediction));
      ui.results.append(row);
    }
    if (resultSkipped.length)
      ui.results.append(element("p", "composer-skipped", "Not reached: " + resultSkipped.join(", ")));
  }

  function renderRun() {
    ui.run.replaceChildren(icon("play"), document.createTextNode(running ? " Running workflow" : " Run workflow"));
    ui.run.disabled = running || !models.some(entry => entry.id === model) || !model;
    ui.stop.hidden = !running;
    ui.status.textContent = runStatus || modelError && !models.length || "";
    ui.status.classList.toggle("is-error", Boolean(runStatus && /failed|invalid|error|stopped/i.test(runStatus)));
    renderResults();
  }

  function render() {
    try {
      layout = layoutTree(root);
      overviewLayout = layoutTree(root, {compact: true});
      renderCount();
      drawMaps();
    } catch (error) {
      layout = null;
      announce(error.message, true);
    }
    renderInspector();
    renderJson();
    renderTabs();
    renderRun();
    ui.undo.disabled = !undoStack.length;
  }

  function fitMap(viewport, tree, isOverview = false) {
    if (isOverview) tree = overviewLayout;
    if (!viewport?.clientWidth || !tree) return;
    const padding = 36;
    const scale = Math.min(1, (viewport.clientWidth - padding) / tree.width,
      viewport.clientHeight > 120 ? (viewport.clientHeight - padding) / tree.height : 1);
    if (isOverview) overviewZoom = Math.max(0.001, scale);
    else zoom = Math.max(0.3, scale);
    drawMaps();
  }

  function download(name, content) {
    const url = URL.createObjectURL(new Blob([content], {type: "application/json"}));
    const anchor = element("a");
    anchor.href = url;
    anchor.download = name;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function importText(source) {
    try {
      const parsed = validateCompositionConfig(JSON.parse(source));
      const previous = JSON.parse(JSON.stringify(root));
      root = draftFromCanonical(parsed.root);
      model = parsed.model;
      stateText = typeof parsed.state === "string" ? parsed.state : JSON.stringify(parsed.state, null, 2);
      jsonState = typeof parsed.state !== "string";
      ui.state.value = stateText;
      ui.jsonToggle.checked = jsonState;
      updateModels();
      rememberSnapshot(previous);
      selectedId = root.id;
      staleResults = resultRows.length > 0;
      jsonDirty = false;
      persist();
      render();
      announce("Workflow imported.");
      return true;
    } catch (error) {
      ui.jsonError.hidden = false;
      ui.jsonError.textContent = error.message;
      ui.jsonEditor.setAttribute("aria-invalid", "true");
      announce(error.message, true);
      return false;
    }
  }

  async function runWorkflow() {
    if (running) return;
    let config;
    try { config = currentConfig(); }
    catch (error) { setStatus(error.message, true); return; }
    running = true;
    controller = new AbortController();
    resultRows = [];
    resultSkipped = [];
    staleResults = false;
    setStatus("Starting workflow…");
    renderRun();
    try {
      const result = await executeWorkflow(config, async (request, signal) => {
        const response = await fetch("/v1/systemone", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(request),
          signal
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok)
          throw new Error(typeof body.detail === "string" ? body.detail : "Request failed (" + response.status + ").");
        return body;
      }, {
        signal: controller.signal,
        onProgress: item => {
          resultRows.push({id: item.id, type: item.type, prediction: item.prediction});
          setStatus("Decision " + item.completed + " · " + item.id + " → " + item.prediction);
          renderResults();
        }
      });
      resultSkipped = result.skipped;
      setStatus("Complete · " + result.visited.length + " decisions evaluated.");
    } catch (error) {
      setStatus(error.name === "AbortError" ? "Workflow stopped." : error.message, error.name !== "AbortError");
    } finally {
      running = false;
      controller = null;
      renderRun();
    }
  }

  function addZoomControl(name, label, amount) {
    const button = element("button", "composer-map-button");
    button.type = "button";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.append(icon(name));
    button.addEventListener("click", () => {
      if (name === "fit") fitMap(ui.mapScroll, layout);
      else {
        zoom = Math.max(0.3, Math.min(1.8, zoom + amount));
        drawMaps();
      }
    });
    ui.zoom.append(button);
  }

  addZoomControl("zoomOut", "Zoom out", -0.12);
  addZoomControl("fit", "Fit workflow", 0);
  addZoomControl("zoomIn", "Zoom in", 0.12);

  container.addEventListener("click", event => {
    const control = event.target.closest("[data-action]");
    if (!control) return;
    const action = control.dataset.action;
    if (action === "add") {
      const entry = walkTree(root).find(item => item.node.id === selectedId);
      try {
        const branch = branchLabels(entry.node)[0];
        const child = createNode(root);
        remember();
        attachNode(root, entry.node.id, branch, child);
        selectedId = child.id;
        persist();
        render();
      } catch (error) { announce(error.message, true); }
    } else if (action === "example") {
      remember();
      root = exampleRoot();
      selectedId = root.id;
      stateText = "A customer reports an urgent export issue and asks for a specialist.";
      jsonState = false;
      ui.state.value = stateText;
      ui.jsonToggle.checked = false;
      resultRows = [];
      resultSkipped = [];
      staleResults = false;
      runStatus = "";
      persist();
      render();
    } else if (action === "undo") {
      const previous = undoStack.pop();
      if (previous) {
        root = previous;
        selectedId = walkTree(root).some(item => item.node.id === selectedId) ? selectedId : root.id;
        staleResults = resultRows.length > 0;
        persist();
        render();
      }
    } else if (action === "download") {
      try { download("jevembed-workflow.json", JSON.stringify(currentConfig(), null, 2) + "\n"); }
      catch (error) { announce(error.message, true); }
    } else if (action === "schema") {
      download("jevembed-workflow.schema.json", JSON.stringify(COMPOSITION_SCHEMA, null, 2) + "\n");
    } else if (action === "import") {
      $("[data-file]").click();
    } else if (action === "apply-json") {
      if (importText(ui.jsonEditor.value)) jsonDirty = false;
    } else if (action === "copy") {
      navigator.clipboard.writeText(ui.jsonEditor.value)
        .then(() => announce("Workflow JSON copied."))
        .catch(error => announce(error.message, true));
    } else if (action === "fit-overview") {
      fitMap(ui.overviewScroll, layout, true);
    } else if (action === "run") {
      runWorkflow();
    } else if (action === "stop") {
      controller?.abort();
    }
  });

  for (const tab of ui.tabs) tab.addEventListener("click", () => {
    mode = tab.dataset.tab;
    renderTabs();
    if (mode === "overview") requestAnimationFrame(() => fitMap(ui.overviewScroll, layout, true));
    drawMaps();
  });

  ui.model.addEventListener("change", () => {
    model = ui.model.value;
    staleResults = resultRows.length > 0;
    persist();
    renderRun();
  });
  ui.state.value = stateText;
  ui.jsonToggle.checked = jsonState;
  ui.state.addEventListener("input", () => {
    stateText = ui.state.value;
    staleResults = resultRows.length > 0;
    persist();
    renderJson();
    renderRun();
  });
  ui.jsonToggle.addEventListener("change", () => {
    if (ui.jsonToggle.checked) {
      const current = ui.state.value;
      try { ui.state.value = JSON.stringify(JSON.parse(current), null, 2); }
      catch { ui.state.value = JSON.stringify(current, null, 2); }
    } else {
      try {
        const current = JSON.parse(ui.state.value);
        ui.state.value = typeof current === "string" ? current : JSON.stringify(current, null, 2);
      } catch {}
    }
    jsonState = ui.jsonToggle.checked;
    stateText = ui.state.value;
    staleResults = resultRows.length > 0;
    persist();
    renderJson();
    renderRun();
  });
  ui.jsonEditor.addEventListener("input", () => {
    jsonDirty = true;
    ui.jsonError.hidden = true;
    ui.jsonEditor.removeAttribute("aria-invalid");
  });
  $("[data-file]").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    importText(await file.text());
    event.target.value = "";
  });
  $("[data-import-text]").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (file) importText(await file.text());
    event.target.value = "";
  });

  const observer = new ResizeObserver(() => {
    if (mode === "overview") requestAnimationFrame(() => fitMap(ui.overviewScroll, layout, true));
  });
  observer.observe(ui.overviewScroll);

  updateModels();
  render();

  return {
    show() {
      container.hidden = false;
      requestAnimationFrame(() => {
        if (mode === "overview") fitMap(ui.overviewScroll, layout, true);
      });
    },
    hide() {
      endPointerDrag();
      container.hidden = true;
      controller?.abort();
    },
    setModels(entries, error = "") {
      models = entries;
      modelError = error;
      updateModels();
      persist();
    }
  };
}
