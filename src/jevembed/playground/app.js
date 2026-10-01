import {COMPOSITION_SCHEMA, PRESETS, buildQuestion, buildCompositionRequest, buildRequest, comparison, hardPrediction, labelsForQuestion, preferredModelId, scoreSummary, simpleCriteriaText, trainingRecord} from "./logic.mjs?v=20261001";
import {initGames} from "./games.js?v=20261001";

const $ = id => document.getElementById(id);
const ui = {
  appShell: $("app-shell"), navToggle: $("nav-toggle"), moduleList: $("module-list"),
  form: $("playground-form"), preset: $("preset"), model: $("model"), modelMessage: $("model-message"),
  state: $("state"), jsonState: $("json-state"), instructions: $("instructions"),
  criteria: $("criteria"), criteriaLabel: $("criteria-label"), criteriaHint: $("criteria-hint"),
  advanced: $("advanced"), advancedWrap: $("advanced-wrap"),
  compareModel: $("compare-model"), compareModelWrap: $("compare-model-wrap"),
  modelLabel: $("model-label"), moduleTitle: $("module-title"), introLede: $("intro-lede"),
  resultsTitle: $("results-title"), resultsSubtitle: $("results-subtitle"),
  run: $("run"), runLabel: $("run-label"), stop: $("stop"), status: $("run-status"), empty: $("empty-state"),
  content: $("results-content"), resultList: $("result-list"), stale: $("stale-note"),
  viewSwitch: $("view-switch"), visualView: $("visual-view"), jsonView: $("json-view"),
  visualResults: $("visual-results"), jsonResults: $("json-results"),
  results: document.querySelector(".results"), busy: $("busy-indicator"), busyTitle: $("busy-title"), busyContext: $("busy-context"),
  previous: $("previous-note"), correct: $("correct-answer"), export: $("export"),
  decisionWorkspace: $("decision-workspace"), gameLab: $("game-lab"), gameModel: $("game-model"),
  compositionWorkspace: $("composition-workspace"), compositionModel: $("composition-model"), compositionState: $("composition-state"),
  compositionJsonState: $("composition-json-state"), compositionSteps: $("composition-steps"), compositionStepCount: $("composition-step-count"),
  compositionMap: $("composition-map"), compositionLinks: $("composition-links"), compositionRoot: $("composition-root"),
  compositionStatus: $("composition-status"), compositionJson: $("composition-json"), addStep: $("add-step"), runComposition: $("run-composition"),
  copyComposition: $("copy-composition"), downloadComposition: $("download-composition"), downloadSchema: $("download-schema"), compositionTemplate: $("composition-template"),
  compositionResponseWrap: $("composition-response-wrap"), compositionResponse: $("composition-response"), copyCompositionResponse: $("copy-composition-response"),
};

const MODULES = [
  {id: "playground", label: "Playground", href: "#/playground"},
  {id: "compare", label: "Compare models", href: "#/compare"},
  {id: "compose", label: "Compose workflow", href: "#/compose"},
  {id: "games", label: "Game Lab", href: "#/games"},
];
const mobileNavQuery = window.matchMedia("(max-width: 900px)");
const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
const animations = new Set();

let currentRun = null;
let previousRun = null;
let secondRun = null;
let lastCompareModel = null;
let secondError = null;
let controller = null;
let activeSequence = 0;
let resultView = "visual";
let navCollapsed = false;
let mobileNavOpen = false;
let activePage = "playground";
const pages = {playground: null, compare: null};
let routeInitialized = false;
let modelLoadError = null;
let modelCount = 0;
let modelsLoaded = false;
const comboboxes = new Map();
const games = initGames();
let openCombobox = null;
let lastPointerPosition = null;
let compositionResizeObserver = null;
const defaultCompositionSteps = () => [
  {id: "route", type: "choice", instructions: "Which team should handle this request?", criteria: "support: General support\nengineering: Technical issue\naccount: Billing or account", advanced: false},
  {id: "severity", type: "score", instructions: "How severe is the request?", criteria: "Low impact\nNeeds attention\nBlocking", advanced: false},
  {id: "escalate", type: "noul", instructions: "Should a human agent review this request?", criteria: "", advanced: false},
];
let compositionSteps = defaultCompositionSteps();
let compositionRequest = null;
let compositionResponse = null;
let compositionController = null;

function routeFromHash() { return location.hash === "#/compare" ? "compare" :
  location.hash === "#/compose" ? "compose" : location.hash === "#/games" ? "games" : "playground"; }
function isCompare() { return activePage === "compare"; }
function isGame() { return activePage === "games"; }
function isCompose() { return activePage === "compose"; }

function cancelAnimations() {
  for (const animation of animations) animation.cancel();
  animations.clear();
}

function animate(element, keyframes, options) {
  if (reducedMotionQuery.matches || typeof element.animate !== "function") return;
  const animation = element.animate(keyframes, options);
  animations.add(animation);
  const forget = () => animations.delete(animation);
  animation.addEventListener("finish", forget, {once: true});
  animation.addEventListener("cancel", forget, {once: true});
}

function updateModuleNav() {
  const mobile = mobileNavQuery.matches;
  const expanded = mobile ? mobileNavOpen : !navCollapsed;
  ui.appShell.classList.toggle("rail-collapsed", !mobile && navCollapsed);
  ui.appShell.classList.toggle("mobile-nav-open", mobile && mobileNavOpen);
  ui.navToggle.setAttribute("aria-expanded", String(expanded));
  const toggleLabel = mobile ? (expanded ? "Close modules" : "Open modules") : (expanded ? "Collapse modules" : "Expand modules");
  ui.navToggle.setAttribute("aria-label", toggleLabel);
  ui.navToggle.title = toggleLabel;
}

function initializeModules() {
  for (const module of MODULES) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.className = "module-link";
    link.href = module.href;
    link.setAttribute("aria-label", module.label);
    link.dataset.module = module.id;
    link.addEventListener("click", event => {
      if (routeFromHash() === module.id) event.preventDefault();
      if (mobileNavQuery.matches) {
        mobileNavOpen = false;
        updateModuleNav();
        if (event.defaultPrevented) ui.navToggle.focus();
      }
    });
    const mark = appendText(link, "span", "module-mark", "");
    mark.setAttribute("aria-hidden", "true");
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("fill", "none");
    icon.setAttribute("stroke", "currentColor");
    icon.setAttribute("stroke-width", "1.8");
    if (module.id === "playground") {
      for (const [x, y] of [[3, 3], [13, 3], [3, 13], [13, 13]]) {
        const cell = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        for (const [name, value] of Object.entries({x, y, width: 8, height: 8, rx: 1.5})) cell.setAttribute(name, String(value));
        icon.append(cell);
      }
    } else if (module.id === "compare") {
      for (const d of ["M3 7h6v10H3z", "M15 7h6v10h-6z", "M9 12h6", "m12 9 3 3-3 3"]) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        icon.append(path);
      }
    } else {
      for (const d of ["M7 8h10a4 4 0 0 1 3.9 3.1l1.1 5a2 2 0 0 1-3.1 2.1L16 16H8l-2.9 2.2A2 2 0 0 1 2 16.1l1.1-5A4 4 0 0 1 7 8Z", "M7 12h4", "M9 10v4", "M16 11h.01", "M18 13h.01"]) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        icon.append(path);
      }
    }
    mark.append(icon);
    appendText(link, "span", "module-label", module.label);
    item.append(link);
    ui.moduleList.append(item);
  }
  updateModuleNav();
}

function snapshot() {
  return {
    preset: ui.preset.value, model: ui.model.value, compareModel: ui.compareModel.value,
    state: ui.state.value, jsonState: ui.jsonState.checked, instructions: ui.instructions.value,
    criteria: ui.criteria.value, advanced: ui.advanced.checked,
    currentRun, previousRun, secondRun, secondError, lastCompareModel, resultView,
    correction: ui.correct.value, correctionOpen: document.querySelector(".correction").open,
    status: ui.status.textContent, statusError: ui.status.classList.contains("error"),
  };
}

function restore(page) {
  const saved = pages[page];
  const data = saved || {...snapshot(), currentRun: null, previousRun: null, secondRun: null,
    secondError: null, lastCompareModel: null, resultView: "visual", correction: "",
    correctionOpen: false, status: "", statusError: false};
  ui.preset.value = data.preset;
  ui.model.value = data.model;
  if (!ui.model.value && ui.model.options.length && ui.model.options[0].value) ui.model.selectedIndex = 0;
  ui.compareModel.value = data.compareModel;
  if (!ui.compareModel.value && ui.compareModel.options.length > 1) ui.compareModel.selectedIndex = 1;
  if (page === "compare" && !saved && modelCount > 1 && ui.compareModel.value === ui.model.value) {
    const alternate = [...ui.compareModel.options].find(entry => entry.value && entry.value !== ui.model.value);
    if (alternate) ui.compareModel.value = alternate.value;
  }
  ui.state.value = data.state;
  ui.jsonState.checked = data.jsonState;
  ui.instructions.value = data.instructions;
  ui.criteria.value = data.criteria;
  ui.advanced.checked = data.advanced;
  currentRun = data.currentRun;
  previousRun = data.previousRun;
  secondRun = data.secondRun;
  secondError = data.secondError;
  lastCompareModel = data.lastCompareModel;
  resultView = data.resultView;
  updateCriteriaPresentation();
  populateCorrections();
  ui.correct.value = data.correction;
  ui.export.disabled = !ui.correct.value;
  document.querySelector(".correction").open = data.correctionOpen;
  setStatus(data.status, data.statusError);
  render();
}

function applyRoute(focus = false) {
  closeCombobox();
  if (location.hash && !["#/playground", "#/compare", "#/compose", "#/games"].includes(location.hash)) {
    history.replaceState(null, "", `${location.pathname}${location.search}#/playground`);
  }
  const page = routeFromHash();
  if (page === activePage && routeInitialized) return;
  if (controller) {
    activeSequence++;
    controller.abort();
    controller = null;
    setRunning(false);
    setStatus("Stopped waiting after changing pages. The server may still finish the request.");
  }
  if (routeInitialized) {
    if (activePage === "games") games.hide();
    else pages[activePage] = snapshot();
  }
  activePage = page;
  routeInitialized = true;
  ui.appShell.classList.toggle("compare-page", isCompare());
  ui.appShell.classList.toggle("game-page", isGame());
  ui.appShell.classList.toggle("compose-page", isCompose());
  ui.decisionWorkspace.hidden = isGame() || isCompose();
  ui.gameLab.hidden = !isGame();
  ui.compositionWorkspace.hidden = !isCompose();
  for (const link of ui.moduleList.querySelectorAll(".module-link")) {
    if (link.dataset.module === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  ui.moduleTitle.textContent = isGame() ? "Game Lab" : isCompose() ? "Compose workflow" : isCompare() ? "Compare models" : "Playground";
  document.title = `${ui.moduleTitle.textContent} · JevEmbed`;
  ui.introLede.textContent = isGame() ? "See how a model plays four tiny pixel games, one turn at a time." :
    isCompose() ? "Combine Jev primitives into one reusable decision workflow." : isCompare() ? "See how two models answer the same question." : "Try a rule and see how the model weighs each option.";
  if (isGame()) {
    games.show();
    if (focus) {
      const heading = document.querySelector(".game-lab-heading h2");
      heading.tabIndex = -1;
      heading.focus({preventScroll: true});
    }
    return;
  }
  if (isCompose()) {
    renderComposition();
    return;
  }
  ui.modelLabel.textContent = isCompare() ? "Model A" : "Model";
  ui.compareModelWrap.hidden = !isCompare();
  ui.resultsTitle.textContent = isCompare() ? "Model comparison" : "The decision";
  ui.resultsSubtitle.hidden = !isCompare();
  if (isCompare()) {
    ui.visualResults.tabIndex = 0;
    ui.visualResults.setAttribute("aria-label", "Model comparison visual results");
  } else {
    ui.visualResults.removeAttribute("tabindex");
    ui.visualResults.removeAttribute("aria-label");
  }
  restore(page);
  setRunning(false);
  updateModelMessage();
  if (focus) {
    const heading = $("editor-title");
    heading.tabIndex = -1;
    heading.focus({preventScroll: true});
  }
}

function option(value, label) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = label;
  return element;
}

function closeCombobox() {
  if (!openCombobox) return;
  openCombobox.menu.hidden = true;
  openCombobox.button.setAttribute("aria-expanded", "false");
  openCombobox.button.removeAttribute("aria-activedescendant");
  openCombobox = null;
}

function refreshComboboxes() {
  for (const {select, button, value} of comboboxes.values()) {
    const label = select.selectedOptions[0]?.textContent || select.options[0]?.textContent || "Choose an option";
    value.textContent = label;
    button.title = label;
    button.disabled = select.disabled || (select === ui.correct ? !currentRun : select !== ui.preset && !modelCount);
  }
}

function positionCombobox(data) {
  const rect = data.button.getBoundingClientRect();
  const edge = 8;
  const viewportWidth = window.visualViewport?.width || window.innerWidth;
  const viewportHeight = window.visualViewport?.height || window.innerHeight;
  const width = Math.min(viewportWidth - edge * 2, rect.width);
  const below = viewportHeight - rect.bottom - edge;
  const above = rect.top - edge;
  const opensAbove = below < 180 && above > below;
  const available = Math.max(80, opensAbove ? above : below);
  const height = Math.min(320, available, data.menu.scrollHeight);
  data.menu.style.width = `${width}px`;
  data.menu.style.maxHeight = `${Math.min(320, available)}px`;
  data.menu.style.left = `${Math.max(edge, Math.min(rect.left, viewportWidth - width - edge))}px`;
  data.menu.style.top = `${opensAbove ? Math.max(edge, rect.top - height - 5) : rect.bottom + 5}px`;
}

function repositionOpenCombobox() {
  if (!openCombobox) return;
  const rect = openCombobox.button.getBoundingClientRect();
  const viewportHeight = window.visualViewport?.height || window.innerHeight;
  if (rect.bottom <= 0 || rect.top >= viewportHeight) closeCombobox();
  else positionCombobox(openCombobox);
}

function activateComboboxOption(data, index) {
  const options = [...data.menu.children];
  if (!options.length) return;
  data.activeIndex = (index + options.length) % options.length;
  options.forEach((node, nodeIndex) => node.classList.toggle("is-active", nodeIndex === data.activeIndex));
  data.button.setAttribute("aria-activedescendant", options[data.activeIndex].id);
  options[data.activeIndex].scrollIntoView({block: "nearest"});
}

function chooseComboboxOption(data, index) {
  const changed = data.select.selectedIndex !== index;
  data.select.selectedIndex = index;
  closeCombobox();
  refreshComboboxes();
  if (changed) {
    data.select.dispatchEvent(new Event("input", {bubbles: true}));
    data.select.dispatchEvent(new Event("change", {bubbles: true}));
  }
  data.button.focus({preventScroll: true});
}

function openComboboxMenu(data) {
  if (data.button.disabled) return;
  closeCombobox();
  data.menu.replaceChildren();
  [...data.select.options].forEach((entry, index) => {
    const item = document.createElement("div");
    item.id = `${data.select.id}-option-${index}`;
    item.className = "combo-option";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(index === data.select.selectedIndex));
    item.textContent = entry.textContent;
    item.addEventListener("pointermove", event => {
      const moved = lastPointerPosition ?
        event.clientX !== lastPointerPosition.x || event.clientY !== lastPointerPosition.y :
        Boolean(event.movementX || event.movementY);
      if (moved) activateComboboxOption(data, index);
    });
    item.addEventListener("click", () => chooseComboboxOption(data, index));
    data.menu.append(item);
  });
  data.menu.hidden = false;
  data.button.setAttribute("aria-expanded", "true");
  openCombobox = data;
  positionCombobox(data);
  activateComboboxOption(data, Math.max(0, data.select.selectedIndex));
}

function initializeComboboxes() {
  for (const select of [ui.preset, ui.model, ui.compareModel, ui.correct, ui.gameModel, ui.compositionModel]) {
    const wrap = select.closest(".select-wrap");
    const label = document.querySelector(`label[for="${select.id}"]`);
    const button = document.createElement("button");
    const menu = document.createElement("div");
    button.type = "button";
    button.id = `${select.id}-trigger`;
    button.className = "combo-trigger";
    button.setAttribute("role", "combobox");
    button.setAttribute("aria-haspopup", "listbox");
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", `${select.id}-listbox`);
    if (select.hasAttribute("aria-describedby")) button.setAttribute("aria-describedby", select.getAttribute("aria-describedby"));
    label.id ||= `${select.id}-label`;
    label.htmlFor = button.id;
    button.setAttribute("aria-labelledby", label.id);
    const value = appendText(button, "span", "combo-value", "");
    menu.id = `${select.id}-listbox`;
    menu.className = "combo-menu";
    if (select === ui.correct) menu.classList.add("is-dark");
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-labelledby", label.id);
    menu.hidden = true;
    document.body.append(menu);
    wrap.append(button);
    wrap.classList.add("is-enhanced");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");
    const data = {select, button, value, menu, activeIndex: 0};
    comboboxes.set(select.id, data);
    button.addEventListener("click", () => openCombobox === data ? closeCombobox() : openComboboxMenu(data));
    button.addEventListener("keydown", event => {
      const key = event.key;
      if (key === "Tab") { closeCombobox(); return; }
      if (key === "Escape") {
        if (openCombobox === data) { event.preventDefault(); closeCombobox(); }
        return;
      }
      if (["ArrowDown", "ArrowUp", "Home", "End", "Enter", " "].includes(key)) event.preventDefault();
      else return;
      if (openCombobox !== data) {
        openComboboxMenu(data);
        if (key === "ArrowDown") activateComboboxOption(data, data.activeIndex + 1);
        else if (key === "ArrowUp") activateComboboxOption(data, data.activeIndex - 1);
        else if (key === "Home") activateComboboxOption(data, 0);
        else if (key === "End") activateComboboxOption(data, data.menu.children.length - 1);
      } else if (key === "ArrowDown") activateComboboxOption(data, data.activeIndex + 1);
      else if (key === "ArrowUp") activateComboboxOption(data, data.activeIndex - 1);
      else if (key === "Home") activateComboboxOption(data, 0);
      else if (key === "End") activateComboboxOption(data, data.menu.children.length - 1);
      else if (key === "Enter" || key === " ") chooseComboboxOption(data, data.activeIndex);
    });
    select.addEventListener("change", refreshComboboxes);
  }
  document.addEventListener("pointerdown", event => {
    if (openCombobox && !openCombobox.button.contains(event.target) && !openCombobox.menu.contains(event.target)) closeCombobox();
  });
  document.addEventListener("pointermove", event => {
    lastPointerPosition = {x: event.clientX, y: event.clientY};
  });
  window.addEventListener("scroll", event => {
    if (openCombobox && !openCombobox.menu.contains(event.target)) repositionOpenCombobox();
  }, true);
  window.addEventListener("resize", closeCombobox);
  window.visualViewport?.addEventListener("resize", closeCombobox);
  window.visualViewport?.addEventListener("scroll", repositionOpenCombobox);
  refreshComboboxes();
}

function setStatus(message, error = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("error", error);
}

function updateModelMessage() {
  if (!modelsLoaded) ui.modelMessage.textContent = "Loading models…";
  else if (modelLoadError) ui.modelMessage.textContent = modelLoadError;
  else if (!modelCount) ui.modelMessage.textContent = "Start the server with a registered model to use the playground.";
  else if (isCompare() && modelCount < 2) ui.modelMessage.textContent = "Compare models needs two available models. Add another model to enable comparison.";
  else ui.modelMessage.textContent = "";
}

function presetType() { return PRESETS[ui.preset.value].type; }

function updateCriteriaPresentation() {
  const type = presetType();
  ui.criteriaLabel.textContent = type === "choice" ? "Choices" : type === "score" ? "Levels" : "Criteria";
  ui.criteriaHint.textContent = ui.advanced.checked ?
    (type === "choice" ? "A JSON object of labels and descriptions." : type === "score" ? "A JSON array of 2–10 ordered levels." : 'Use null, or {"true": "…", "false": "…"}.') :
    (type === "choice" ? "One per line: label or label: description" : type === "score" ? "One level per line, from lowest to highest." : "Optional: turn on JSON criteria to describe true and false.");
  ui.criteria.placeholder = type === "noul" ? (ui.advanced.checked ? 'null' : "No criteria needed") : "";
  ui.criteria.disabled = type === "noul" && !ui.advanced.checked;
  ui.criteria.hidden = type === "noul" && !ui.advanced.checked;
  ui.criteria.rows = type === "noul" ? 2 : 3;
}

function setPreset(key) {
  const preset = PRESETS[key];
  ui.state.value = preset.state;
  ui.jsonState.checked = Boolean(preset.jsonState);
  ui.instructions.value = preset.instructions;
  ui.criteria.value = preset.criteria;
  ui.advanced.checked = false;
  updateCriteriaPresentation();
  updateStaleness();
}

function draft(model = ui.model.value) {
  return {model, state: ui.state.value, jsonState: ui.jsonState.checked,
    type: presetType(), instructions: ui.instructions.value,
    criteria: ui.criteria.value, advanced: ui.advanced.checked};
}

function currentRequest() { return buildRequest(draft()); }

function updateStaleness() {
  if (!currentRun) return;
  let stale = true;
  try { stale = JSON.stringify(currentRequest()) !== JSON.stringify(currentRun.request); }
  catch { /* Invalid edits are still stale. */ }
  if (isCompare()) stale ||= ui.compareModel.value !== lastCompareModel;
  ui.stale.hidden = !stale;
}

function setRunning(running) {
  if (running) closeCombobox();
  if (running) cancelAnimations();
  for (const control of ui.form.querySelectorAll("input,select,textarea,button")) control.disabled = running;
  ui.run.disabled = running || !ui.model.value || (isCompare() && ui.model.options.length < 2);
  ui.run.classList.toggle("is-running", running);
  ui.runLabel.textContent = running ? "Running…" : isCompare() ? "Compare models" : "Run example";
  ui.stop.disabled = !running;
  ui.stop.hidden = !running;
  ui.results.setAttribute("aria-busy", String(running));
  ui.results.classList.toggle("is-busy", running);
  ui.results.classList.toggle("is-first-run", running && !currentRun);
  ui.busy.hidden = !running;
  ui.busyContext.hidden = !currentRun;
  ui.empty.hidden = running || Boolean(currentRun);
  if (!running) {
    ui.criteria.disabled = presetType() === "noul" && !ui.advanced.checked;
    ui.compareModelWrap.hidden = !isCompare();
  }
  refreshComboboxes();
}

function formatLabel(label, question) {
  if (question.type === "noul") return label === "true" ? "Yes" : "No";
  if (question.type === "score") return `Level ${label}`;
  return label;
}

function formatPercent(value) { return `${(Number(value) * 100).toFixed(1)}%`; }
function formatElapsed(ms) { return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`; }

function appendText(parent, tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  parent.append(node);
  return node;
}

function probabilityEntries(answer, question) {
  if (question.type === "noul") return [["true", Number(answer.noul)], ["false", 1 - Number(answer.noul)]];
  return labelsForQuestion(question).map(label => [label, Number(answer.probabilities?.[label] ?? 0)]);
}

function renderCard(run, heading, corrected) {
  const card = document.createElement("article");
  card.className = "result-card";
  const question = run.request.questions.decision;
  const prediction = hardPrediction(run.answer);
  appendText(card, "h3", "", heading);
  const line = appendText(card, "div", "decision-line", "");
  if (question.type === "score") {
    appendText(line, "span", "decision", scoreSummary(run.answer, question).value);
  } else appendText(line, "span", "decision", formatLabel(prediction, question));
  const latency = appendText(line, "span", "latency", formatElapsed(run.elapsed));
  latency.title = "Wall time, including network, model loading and cache work.";
  latency.setAttribute("aria-label", `${formatElapsed(run.elapsed)} wall time, including network, model loading and cache work`);
  if (question.type === "score") appendText(card, "div", "score-detail", `Most likely: ${scoreSummary(run.answer, question).mostLikely}`);
  const bars = appendText(card, "div", "distribution", "");
  for (const [label, probability] of probabilityEntries(run.answer, question)) {
    const row = appendText(bars, "div", `bar-row${label === prediction ? " leading" : ""}`, "");
    const labelRow = appendText(row, "div", "bar-label", "");
    let display = formatLabel(label, question);
    if (question.type === "score") {
      const description = question.criteria[Number(label)];
      display += ` · ${typeof description === "string" ? description : JSON.stringify(description)}`;
    }
    appendText(labelRow, "span", "", display);
    appendText(labelRow, "strong", "", formatPercent(probability));
    const track = appendText(row, "div", "bar-track", "");
    const fill = appendText(track, "div", "bar-fill", "");
    fill.style.width = `${Math.min(100, Math.max(0, probability * 100))}%`;
  }
  if (corrected) appendText(card, "div", "result-verdict", prediction === corrected ? "Matches your correction" : "Differs from your correction");
  return card;
}

function populateCorrections() {
  const previous = ui.correct.value;
  ui.correct.replaceChildren(option("", "Select an answer"));
  if (!currentRun) { refreshComboboxes(); return; }
  const question = currentRun.request.questions.decision;
  for (const label of labelsForQuestion(question)) ui.correct.append(option(label, formatLabel(label, question)));
  ui.correct.value = labelsForQuestion(question).includes(previous) ? previous : "";
  ui.export.disabled = !ui.correct.value;
  refreshComboboxes();
}

function setResultView(view, transition = false) {
  const changed = view !== resultView;
  if (changed) cancelAnimations();
  resultView = view;
  ui.visualView.setAttribute("aria-pressed", String(view === "visual"));
  ui.jsonView.setAttribute("aria-pressed", String(view === "json"));
  ui.visualResults.hidden = view !== "visual";
  ui.jsonResults.hidden = view !== "json";
  if (changed && transition && currentRun) {
    const target = view === "visual" ? ui.visualResults : ui.jsonResults;
    animate(target, [{opacity: 0, transform: "translateY(4px)"}, {opacity: 1, transform: "translateY(0)"}],
      {duration: 180, easing: "cubic-bezier(.2, .75, .2, 1)"});
  }
}

function animateResults() {
  if (resultView === "json") {
    for (const [index, card] of [...ui.jsonResults.querySelectorAll(".raw-card")].entries()) {
      animate(card, [{opacity: 0, transform: "translateY(9px)"}, {opacity: 1, transform: "translateY(0)"}],
        {duration: 300, delay: Math.min(index, 2) * 75, easing: "cubic-bezier(.2, .75, .2, 1)", fill: "both"});
    }
    return;
  }
  for (const [index, card] of [...ui.resultList.querySelectorAll(".result-card")].entries()) {
    animate(card, [{opacity: 0, transform: "translateY(12px)"}, {opacity: 1, transform: "translateY(0)"}],
      {duration: 320, delay: Math.min(index, 2) * 80, easing: "cubic-bezier(.2, .75, .2, 1)", fill: "both"});
  }
  for (const [index, bar] of [...ui.resultList.querySelectorAll(".bar-fill")].entries()) {
    animate(bar, [{transform: "scaleX(0)"}, {transform: "scaleX(1)"}],
      {duration: 490, delay: 110 + Math.min(index, 5) * 50,
        easing: "cubic-bezier(.2, .8, .2, 1)", fill: "both"});
  }
}

function renderRaw(run) {
  const card = document.createElement("article");
  card.className = "raw-card";
  const model = run.response.model || run.request.model;
  appendText(card, "h3", "", model);
  const raw = appendText(card, "pre", "raw-json", JSON.stringify(run.response, null, 2));
  raw.tabIndex = 0;
  raw.setAttribute("aria-label", `${model} raw JSON response`);
  return card;
}

function renderSecondError(raw = false) {
  const card = document.createElement("article");
  card.className = raw ? "raw-card result-error" : "result-card result-error";
  appendText(card, "h3", "", lastCompareModel || "Model B");
  appendText(card, "p", "", `Could not complete this model: ${secondError}`);
  return card;
}

function render() {
  cancelAnimations();
  ui.empty.hidden = Boolean(currentRun);
  ui.content.hidden = !currentRun;
  ui.viewSwitch.hidden = !currentRun;
  if (!currentRun) {
    ui.resultList.replaceChildren();
    ui.jsonResults.replaceChildren();
    return;
  }
  const corrected = ui.correct.value;
  ui.resultList.replaceChildren(renderCard(currentRun, currentRun.request.model, corrected));
  if (isCompare() && secondRun) ui.resultList.append(renderCard(secondRun, secondRun.request.model, corrected));
  else if (isCompare() && secondError) ui.resultList.append(renderSecondError());
  ui.jsonResults.replaceChildren(renderRaw(currentRun));
  if (isCompare() && secondRun) ui.jsonResults.append(renderRaw(secondRun));
  else if (isCompare() && secondError) ui.jsonResults.append(renderSecondError(true));
  setResultView(resultView);
  const change = isCompare() ? null : comparison(previousRun, currentRun);
  ui.previous.hidden = !change;
  if (change) {
    const label = formatLabel(change.after, currentRun.request.questions.decision);
    const points = Math.abs(change.delta * 100).toFixed(1);
    const shift = !Number.isFinite(change.delta) ? "" : points === "0.0" ?
      ` Probability of ${label} was unchanged.` :
      ` Probability of ${label} ${change.delta > 0 ? "rose" : "fell"} ${points} points.`;
    ui.previous.textContent = change.changed ?
      `Previous run: ${formatLabel(change.before, currentRun.request.questions.decision)} → ${formatLabel(change.after, currentRun.request.questions.decision)}. The answer changed.${shift}` :
      `Previous run: the answer stayed ${formatLabel(change.after, currentRun.request.questions.decision)}.${shift}`;
  }
  updateStaleness();
}

async function fetchResult(request, signal) {
  const start = performance.now();
  const response = await fetch("/v1/systemone", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request), signal});
  const body = await response.json().catch(error => {
    if (error.name === "AbortError") throw error;
    return {};
  });
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`);
  const answer = body?.answers?.decision;
  if (!answer || answer.type !== request.questions.decision.type) throw new Error("The server returned an unexpected answer.");
  return {request: structuredClone(request), answer, response: body, elapsed: performance.now() - start};
}

async function run(event) {
  event.preventDefault();
  if (controller) return;
  let request;
  try { request = currentRequest(); }
  catch (error) { setStatus(error.message, true); return; }
  const compareModel = isCompare() ? ui.compareModel.value : null;
  if (isCompare() && ui.model.options.length < 2) { setStatus("Compare models needs two available models.", true); return; }
  if (isCompare() && !compareModel) { setStatus("Choose Model B.", true); return; }
  if (compareModel === request.model) { setStatus("Choose different models for A and B.", true); return; }
  const sequence = ++activeSequence;
  controller = new AbortController();
  setRunning(true);
  ui.busyTitle.textContent = compareModel ? "Running first model…" : "Running model…";
  setStatus(compareModel ? "Running first model…" : "Running…");
  try {
    const first = await fetchResult(request, controller.signal);
    if (sequence !== activeSequence) return;
    let second = null;
    let failedSecond = null;
    if (compareModel) {
      ui.busyTitle.textContent = "Running second model…";
      setStatus("Running second model…");
      try { second = await fetchResult({...request, model: compareModel}, controller.signal); }
      catch (error) {
        if (sequence !== activeSequence) return;
        if (error.name === "AbortError") throw error;
        failedSecond = error.message;
        setStatus(`Model A ready. Model B: ${error.message}`, true);
      }
    }
    if (sequence !== activeSequence) return;
    previousRun = currentRun;
    currentRun = first;
    secondRun = second;
    secondError = failedSecond;
    lastCompareModel = compareModel;
    ui.correct.value = "";
    populateCorrections();
    render();
    animateResults();
    if (!ui.status.classList.contains("error")) setStatus(second ? "Both results are ready." : "Result ready.");
  } catch (error) {
    if (sequence !== activeSequence) return;
    setStatus(error.name === "AbortError" ? "Stopped waiting. The server may still finish the request." : error.message, error.name !== "AbortError");
  } finally {
    if (sequence === activeSequence) { controller = null; setRunning(false); }
  }
}

function convertCriteriaMode() {
  const type = presetType();
  try {
    if (ui.advanced.checked) {
      const question = buildQuestion({type, instructions: ui.instructions.value || "Question", criteria: ui.criteria.value, advanced: false});
      ui.criteria.value = JSON.stringify(question.criteria ?? null, null, 2);
    } else {
      const question = buildQuestion({type, instructions: ui.instructions.value || "Question", criteria: ui.criteria.value, advanced: true});
      ui.criteria.value = simpleCriteriaText(question);
    }
    updateCriteriaPresentation(); updateStaleness(); setStatus("");
  } catch (error) { ui.advanced.checked = !ui.advanced.checked; setStatus(error.message, true); }
}

function exportCorrection() {
  if (!currentRun) return;
  try {
    const record = trainingRecord(currentRun, ui.correct.value, `playground-${Date.now()}`, isCompare() ? secondRun : null);
    const blob = new Blob([`${JSON.stringify(record)}\n`], {type: "application/x-ndjson"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = `${record.id}.jsonl`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setStatus("Labeled example downloaded.");
  } catch (error) { setStatus(error.message, true); }
}

async function loadModels() {
  try {
    const response = await fetch("/v1/models");
    if (!response.ok) throw new Error(`Could not load models (${response.status}).`);
    const body = await response.json();
    const models = Array.isArray(body.data) ? body.data.filter(entry => typeof entry.id === "string" && entry.id) : [];
    modelCount = models.length;
    const selectedModel = ui.model.value;
    const selectedCompareModel = ui.compareModel.value;
    const selectedCompositionModel = ui.compositionModel.value;
    ui.model.replaceChildren(...models.map(entry => option(entry.id, entry.id)));
    ui.compareModel.replaceChildren(...models.map(entry => option(entry.id, entry.id)));
    ui.compositionModel.replaceChildren(...models.map(entry => option(entry.id, entry.id)));
    if (!models.length) {
      ui.model.append(option("", "No models available"));
      ui.compareModel.append(option("", "No models available"));
      ui.compositionModel.append(option("", "No models available"));
    } else {
      const preferredModel = preferredModelId(models);
      if (models.some(entry => entry.id === selectedModel)) ui.model.value = selectedModel;
      else ui.model.value = preferredModel;
      if (models.some(entry => entry.id === selectedCompareModel)) ui.compareModel.value = selectedCompareModel;
      else if (models.length > 1) ui.compareModel.selectedIndex = 1;
      if (models.some(entry => entry.id === selectedCompositionModel)) ui.compositionModel.value = selectedCompositionModel;
      else ui.compositionModel.value = preferredModel;
    }
    games.setModels(models.map(entry => entry.id));
  } catch (error) {
    modelLoadError = error.message;
    ui.model.replaceChildren(option("", "Models unavailable"));
    ui.compareModel.replaceChildren(option("", "Models unavailable"));
    ui.compositionModel.replaceChildren(option("", "Models unavailable"));
    games.setModels([]);
  }
  modelsLoaded = true;
  updateModelMessage();
  if (isCompose()) previewComposition();
  setRunning(false);
}

function compositionTypeLabel(type) { return type === "choice" ? "Choice" : type === "score" ? "Score" : "Noul"; }

function setCompositionStatus(message, error = false) {
  ui.compositionStatus.textContent = message;
  ui.compositionStatus.classList.toggle("error", error);
}

function compositionDraft() {
  return {model: ui.compositionModel.value, state: ui.compositionState.value,
    jsonState: ui.compositionJsonState.checked, steps: compositionSteps};
}

function previewComposition() {
  try {
    compositionRequest = buildCompositionRequest(compositionDraft());
    ui.compositionJson.textContent = JSON.stringify(compositionRequest, null, 2);
    ui.copyComposition.disabled = false;
    ui.downloadComposition.disabled = false;
    return compositionRequest;
  } catch (error) {
    compositionRequest = null;
    ui.compositionJson.textContent = error.message;
    ui.copyComposition.disabled = true;
    ui.downloadComposition.disabled = true;
    return null;
  }
}

function compositionInput(step, key, value) {
  step[key] = value;
  previewComposition();
  if (key === "instructions" || key === "criteria" || key === "id") requestAnimationFrame(drawCompositionLinks);
}

function drawCompositionLinks() {
  if (!ui.compositionMap || !ui.compositionRoot || !ui.compositionLinks) return;
  const mapRect = ui.compositionMap.getBoundingClientRect();
  const rootRect = ui.compositionRoot.getBoundingClientRect();
  const width = Math.max(1, ui.compositionMap.clientWidth);
  const height = Math.max(1, ui.compositionMap.clientHeight);
  ui.compositionLinks.setAttribute("viewBox", `0 0 ${width} ${height}`);
  ui.compositionLinks.setAttribute("width", String(width));
  ui.compositionLinks.setAttribute("height", String(height));
  ui.compositionLinks.replaceChildren();
  for (const card of ui.compositionSteps.querySelectorAll(".composition-step")) {
    const rect = card.getBoundingClientRect();
    const rootRight = rootRect.right - mapRect.left;
    const rootCenter = rootRect.left - mapRect.left + rootRect.width / 2;
    const cardLeft = rect.left - mapRect.left;
    const cardCenter = cardLeft + rect.width / 2;
    const desktop = cardLeft >= rootRight - 4;
    const x1 = desktop ? rootRight : rootCenter;
    const y1 = desktop ? rootRect.top - mapRect.top + rootRect.height / 2 : rootRect.bottom - mapRect.top;
    const x2 = desktop ? cardLeft : cardCenter;
    const y2 = desktop ? rect.top - mapRect.top + rect.height / 2 : rect.top - mapRect.top;
    const bend = desktop ? Math.max(24, (x2 - x1) * .42) : Math.max(22, (y2 - y1) * .42);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.classList.add("composition-link");
    path.setAttribute("d", desktop ? `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}` : `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`);
    ui.compositionLinks.append(path);
  }
}

function compositionNodeIcon(type) { return type === "choice" ? "C" : type === "score" ? "S" : "N"; }

function renderComposition() {
  ui.compositionStepCount.textContent = `${compositionSteps.length} ${compositionSteps.length === 1 ? "step" : "steps"}`;
  ui.compositionRoot.replaceChildren();
  const root = document.createElement("article");
  root.className = "composition-root-node";
  root.setAttribute("aria-label", "Shared workflow state");
  appendText(root, "span", "composition-node-kicker", "WORKFLOW ROOT");
  appendText(root, "strong", "composition-root-title", "Shared state");
  const rootState = appendText(root, "span", "composition-root-state", ui.compositionState.value.trim() || "Add a state to begin");
  rootState.title = ui.compositionState.value;
  appendText(root, "span", "composition-root-meta", `${ui.compositionModel.value || "No model"} · ${compositionSteps.length} decisions`);
  ui.compositionRoot.append(root);
  ui.compositionSteps.replaceChildren();
  compositionSteps.forEach((step, index) => {
    const card = document.createElement("article");
    card.className = `composition-step composition-step-${step.type}`;
    card.dataset.index = String(index);
    card.setAttribute("role", "listitem");
    const header = document.createElement("div");
    header.className = "composition-step-header";
    const nodeBadge = appendText(header, "span", "composition-step-number", compositionNodeIcon(step.type));
    nodeBadge.setAttribute("aria-hidden", "true");
    const titleWrap = appendText(header, "div", "composition-step-title-wrap", "");
    appendText(titleWrap, "span", "composition-node-kicker", `STEP ${String(index + 1).padStart(2, "0")}`);
    appendText(titleWrap, "strong", "composition-step-title", `${compositionTypeLabel(step.type)} decision`);
    const controls = appendText(header, "div", "composition-step-controls", "");
    for (const [action, label, glyph] of [["up", "Move up", "↑"], ["down", "Move down", "↓"], ["remove", "Remove", "×"]]) {
      const button = appendText(controls, "button", "step-icon-button", glyph);
      button.type = "button"; button.dataset.action = action; button.title = label; button.setAttribute("aria-label", label);
      button.disabled = action === "up" ? index === 0 : action === "down" ? index === compositionSteps.length - 1 : compositionSteps.length === 1;
      button.addEventListener("click", () => {
        if (action === "remove") compositionSteps.splice(index, 1);
        else { const target = action === "up" ? index - 1 : index + 1; [compositionSteps[index], compositionSteps[target]] = [compositionSteps[target], compositionSteps[index]]; }
        renderComposition();
      });
    }
    card.append(header);
    const row = appendText(card, "div", "composition-step-row", "");
    const idField = appendText(row, "div", "field", "");
    appendText(idField, "label", "", "Step ID").htmlFor = `composition-id-${index}`;
    const idInput = document.createElement("input"); idInput.id = `composition-id-${index}`; idInput.value = step.id; idInput.className = "composition-input"; idInput.type = "text";
    idInput.addEventListener("input", () => compositionInput(step, "id", idInput.value)); idField.append(idInput);
    const typeField = appendText(row, "div", "field", "");
    appendText(typeField, "label", "", "Primitive").htmlFor = `composition-type-${index}`;
    const typeSelect = document.createElement("select"); typeSelect.id = `composition-type-${index}`; typeSelect.className = "composition-input";
    for (const type of ["choice", "score", "noul"]) typeSelect.append(option(type, compositionTypeLabel(type)));
    typeSelect.value = step.type;
    typeSelect.addEventListener("change", () => { step.type = typeSelect.value; step.criteria = step.type === "noul" ? "" : step.type === "choice" ? "support: General support\nother: Other" : "Low impact\nHigh impact"; step.advanced = false; renderComposition(); });
    typeField.append(typeSelect);
    const instructionField = appendText(card, "div", "field", "");
    appendText(instructionField, "label", "", "Question").htmlFor = `composition-instruction-${index}`;
    const instruction = document.createElement("textarea"); instruction.id = `composition-instruction-${index}`; instruction.rows = 2; instruction.value = step.instructions; instruction.className = "composition-input";
    instruction.addEventListener("input", () => compositionInput(step, "instructions", instruction.value)); instructionField.append(instruction);
    const criteriaField = appendText(card, "div", "field composition-criteria-field", "");
    const criteriaLabel = appendText(criteriaField, "label", "", step.type === "choice" ? "Choices" : step.type === "score" ? "Levels" : "Criteria"); criteriaLabel.htmlFor = `composition-criteria-${index}`;
    const advancedLabel = document.createElement("label"); advancedLabel.className = "small-check";
    const advanced = document.createElement("input"); advanced.type = "checkbox"; advanced.checked = step.advanced;
    const advancedText = document.createElement("span"); advancedText.textContent = "JSON criteria";
    advancedLabel.append(advanced, advancedText);
    advanced.addEventListener("change", () => {
      try {
        if (advanced.checked) {
          const question = buildQuestion(step);
          step.criteria = JSON.stringify(question.criteria ?? null, null, 2);
        } else {
          const question = buildQuestion({...step, advanced: true});
          step.criteria = simpleCriteriaText(question);
        }
        step.advanced = advanced.checked;
        renderComposition();
      } catch (error) { advanced.checked = !advanced.checked; setCompositionStatus(error.message, true); }
    });
    criteriaField.append(advancedLabel);
    const criteria = document.createElement("textarea"); criteria.id = `composition-criteria-${index}`; criteria.rows = step.type === "noul" ? 2 : 3; criteria.value = step.criteria; criteria.className = "composition-input"; criteria.disabled = step.type === "noul" && !step.advanced;
    criteria.placeholder = step.type === "noul" ? (step.advanced ? '{"true":"…","false":"…"}' : "No criteria needed") : "";
    criteria.addEventListener("input", () => compositionInput(step, "criteria", criteria.value)); criteriaField.append(criteria);
    card.append(instructionField); card.append(criteriaField);
    ui.compositionSteps.append(card);
  });
  previewComposition();
  if (compositionResponse) {
    ui.compositionResponseWrap.hidden = false;
    ui.compositionResponse.textContent = JSON.stringify(compositionResponse, null, 2);
  }
  requestAnimationFrame(drawCompositionLinks);
}

async function runCompositionWorkflow() {
  if (compositionController) return;
  const request = previewComposition();
  if (!request) { setCompositionStatus(ui.compositionJson.textContent, true); return; }
  compositionController = new AbortController();
  ui.runComposition.disabled = true;
  setCompositionStatus("Running workflow…");
  try {
    const response = await fetch("/v1/systemone", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request), signal: compositionController.signal});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`);
    compositionResponse = body;
    const output = Object.fromEntries(Object.entries(body.answers || {}).map(([id, answer]) => [id, {type: answer.type, prediction: hardPrediction(answer), answer}]));
    ui.compositionJson.textContent = JSON.stringify(request, null, 2);
    setCompositionStatus(`Workflow complete · ${Object.keys(output).length} decisions ready.`);
    ui.compositionResponseWrap.hidden = false;
    ui.compositionResponse.textContent = JSON.stringify({model: body.model, decisions: output, usage: body.usage}, null, 2);
  } catch (error) { setCompositionStatus(error.name === "AbortError" ? "Stopped waiting." : error.message, error.name !== "AbortError"); }
  finally { compositionController = null; ui.runComposition.disabled = false; }
}

function downloadText(filename, value, type = "application/json") {
  const url = URL.createObjectURL(new Blob([value], {type})); const link = document.createElement("a"); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function wireComposition() {
  ui.compositionModel.addEventListener("input", previewComposition);
  ui.compositionModel.addEventListener("change", () => { renderComposition(); });
  ui.compositionState.addEventListener("input", () => { previewComposition(); renderComposition(); });
  ui.compositionJsonState.addEventListener("change", previewComposition);
  ui.addStep.addEventListener("click", () => { compositionSteps.push({id: `decision_${compositionSteps.length + 1}`, type: "choice", instructions: "What should happen next?", criteria: "yes: Proceed\nno: Stop", advanced: false}); renderComposition(); });
  ui.compositionTemplate.addEventListener("click", () => { compositionSteps = defaultCompositionSteps(); ui.compositionState.value = "A customer reports a failed export and asks to speak to a person."; renderComposition(); setCompositionStatus("Escalation template loaded."); });
  ui.runComposition.addEventListener("click", runCompositionWorkflow);
  ui.copyComposition.addEventListener("click", async () => { if (!compositionRequest) return; await navigator.clipboard.writeText(JSON.stringify(compositionRequest, null, 2)); setCompositionStatus("Request JSON copied."); });
  ui.copyCompositionResponse.addEventListener("click", async () => { if (!compositionResponse) return; await navigator.clipboard.writeText(JSON.stringify(compositionResponse, null, 2)); setCompositionStatus("Response JSON copied."); });
  ui.downloadComposition.addEventListener("click", () => { if (compositionRequest) downloadText("jevembed-workflow.json", `${JSON.stringify(compositionRequest, null, 2)}\n`); });
  ui.downloadSchema.addEventListener("click", () => downloadText("jevembed-workflow.schema.json", `${JSON.stringify(COMPOSITION_SCHEMA, null, 2)}\n`));
  ui.compositionState.value = "A customer reports a failed export and asks to speak to a person.";
  renderComposition();
  if (typeof ResizeObserver === "function") {
    compositionResizeObserver = new ResizeObserver(() => { if (isCompose()) drawCompositionLinks(); });
    compositionResizeObserver.observe(ui.compositionMap);
  }
}

ui.preset.addEventListener("change", () => { setPreset(ui.preset.value); setStatus(""); });
ui.advanced.addEventListener("change", convertCriteriaMode);
for (const control of [ui.model, ui.state, ui.jsonState, ui.instructions, ui.criteria, ui.compareModel]) control.addEventListener("input", updateStaleness);
ui.form.addEventListener("submit", run);
ui.stop.addEventListener("click", () => controller?.abort());
ui.correct.addEventListener("change", () => { ui.export.disabled = !ui.correct.value; render(); });
ui.visualView.addEventListener("click", () => setResultView("visual", true));
ui.jsonView.addEventListener("click", () => setResultView("json", true));
ui.export.addEventListener("click", exportCorrection);
ui.navToggle.addEventListener("click", () => {
  if (mobileNavQuery.matches) mobileNavOpen = !mobileNavOpen;
  else navCollapsed = !navCollapsed;
  updateModuleNav();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && mobileNavQuery.matches && mobileNavOpen) {
    mobileNavOpen = false;
    updateModuleNav();
    ui.navToggle.focus();
  }
});
mobileNavQuery.addEventListener("change", () => { mobileNavOpen = false; updateModuleNav(); });
reducedMotionQuery.addEventListener("change", () => { if (reducedMotionQuery.matches) cancelAnimations(); });
window.addEventListener("hashchange", () => applyRoute(true));
window.addEventListener("resize", () => { if (isCompose()) requestAnimationFrame(drawCompositionLinks); });
initializeModules();
setPreset("choice");
initializeComboboxes();
wireComposition();
applyRoute();
loadModels();
