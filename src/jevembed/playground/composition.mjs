import {buildQuestion, labelsForQuestion, validateCompositionConfig, COMPOSITION_LIMITS} from "./logic.mjs?v=20261003-composer8";

export function walkTree(root) {
  const entries = [];
  const pending = [{node: root, parent: null, branch: null, index: 0, depth: 0}];
  const visited = new WeakSet();
  while (pending.length) {
    const entry = pending.pop();
    if (!entry.node || visited.has(entry.node)) throw new Error("Workflow must be a tree.");
    visited.add(entry.node);
    entries.push(entry);
    const branches = Object.entries(entry.node.children || {});
    for (let branchIndex = branches.length - 1; branchIndex >= 0; branchIndex--) {
      const [branch, children] = branches[branchIndex];
      for (let childIndex = children.length - 1; childIndex >= 0; childIndex--)
        pending.push({node: children[childIndex], parent: entry.node, branch, index: childIndex, depth: entry.depth + 1});
    }
  }
  return entries;
}

export function questionForNode(node) {
  if (typeof node.criteria === "string") return buildQuestion(node);
  const {id, children, ...question} = node;
  return question;
}

export function branchLabels(node) {
  return labelsForQuestion(questionForNode(node));
}

export function createNode(root, type = "choice") {
  const ids = new Set(walkTree(root).map(entry => entry.node.id));
  let sequence = 1;
  while (ids.has(type + "_" + sequence)) sequence++;
  const defaults = {
    choice: {instructions: "What should happen next?", criteria: "yes: Proceed\nno: Stop"},
    score: {instructions: "How severe is this issue?", criteria: "Low impact\nNeeds attention\nBlocking"},
    noul: {instructions: "Does this need human review?", criteria: "null"}
  };
  if (!defaults[type]) throw new Error("Unknown decision type.");
  return {id: type + "_" + sequence, type, ...defaults[type], advanced: type === "noul", children: {}};
}

function checkTreeLimits(root) {
  const entries = walkTree(root);
  if (entries.length > COMPOSITION_LIMITS.maxNodes) throw new Error("Workflow has too many nodes.");
  if (entries.some(entry => entry.depth >= COMPOSITION_LIMITS.maxDepth)) throw new Error("Workflow is too deeply nested.");
}

export function attachNode(root, parentId, branch, child) {
  const parent = walkTree(root).find(entry => entry.node.id === parentId)?.node;
  if (!parent) throw new Error("The parent node no longer exists.");
  if (!branchLabels(parent).includes(branch)) throw new Error("Choose a valid result branch.");
  if (walkTree(root).some(entry => entry.node.id === child.id)) throw new Error("Node IDs must be unique.");
  const branches = parent.children || {};
  const previous = Object.hasOwn(branches, branch) ? branches[branch] : [];
  parent.children = Object.fromEntries([...Object.entries(branches).filter(([label]) => label !== branch), [branch, [...previous, child]]]);
  try { checkTreeLimits(root); }
  catch (error) { parent.children = branches; throw error; }
  return child;
}

export function canMoveNode(root, nodeId, parentId, branch) {
  const entries = walkTree(root);
  const source = entries.find(entry => entry.node.id === nodeId);
  const target = entries.find(entry => entry.node.id === parentId);
  if (!source?.parent || !target) return false;
  if (walkTree(source.node).some(entry => entry.node === target.node)) return false;
  return branchLabels(target.node).includes(branch);
}

export function moveNode(root, nodeId, parentId, branch, index) {
  if (!canMoveNode(root, nodeId, parentId, branch))
    throw new Error("A node cannot move into itself, a descendant, or an invalid branch.");
  const entries = walkTree(root);
  const source = entries.find(entry => entry.node.id === nodeId);
  const target = entries.find(entry => entry.node.id === parentId).node;
  const sourceBranches = source.parent.children;
  const targetBranches = target.children || {};
  const sourceSiblings = sourceBranches[source.branch];
  const sameBranch = source.parent === target && source.branch === branch;
  const siblings = (sameBranch ? sourceSiblings : Object.hasOwn(targetBranches, branch) ? targetBranches[branch] : []).filter(node => node !== source.node);
  const destination = index === undefined ? siblings.length : Math.max(0, Math.min(index, siblings.length));
  siblings.splice(destination, 0, source.node);
  const remaining = sourceSiblings.filter(node => node !== source.node);
  source.parent.children = Object.fromEntries(Object.entries(sourceBranches).flatMap(([label, children]) =>
    label === source.branch ? remaining.length ? [[label, remaining]] : [] : [[label, children]]));
  target.children = Object.fromEntries([...Object.entries(target.children || {}).filter(([label]) => label !== branch), [branch, siblings]]);
  try { checkTreeLimits(root); }
  catch (error) { source.parent.children = sourceBranches; target.children = targetBranches; throw error; }
  return source.node;
}

export function removeNode(root, nodeId) {
  const entry = walkTree(root).find(item => item.node.id === nodeId);
  if (!entry?.parent) throw new Error("The root decision cannot be removed.");
  const remaining = entry.parent.children[entry.branch].filter(node => node !== entry.node);
  entry.parent.children = Object.fromEntries(Object.entries(entry.parent.children).flatMap(([label, children]) =>
    label === entry.branch ? remaining.length ? [[label, remaining]] : [] : [[label, children]]));
  return entry.parent.id;
}

export function replaceNode(root, nodeId, replacement) {
  const entry = walkTree(root).find(item => item.node.id === nodeId);
  if (!entry) throw new Error("The selected node no longer exists.");
  if (walkTree(root).some(item => item.node !== entry.node && item.node.id === replacement.id))
    throw new Error("Node IDs must be unique.");
  const existingChildren = entry.node.children || {};
  const allowed = branchLabels(replacement);
  let children = existingChildren;
  if (entry.node.type !== replacement.type) {
    children = {};
    const previousBranches = Object.entries(existingChildren).filter(([, items]) => items.length);
    previousBranches.forEach(([, items], index) => {
      const target = allowed[Math.min(index, allowed.length - 1)];
      children[target] = [...(children[target] || []), ...items];
    });
  } else {
    const incompatible = Object.entries(existingChildren).filter(([label, items]) => items.length && !allowed.includes(label));
    if (incompatible.length)
      throw new Error("Move or remove child nodes on these branches first: " + incompatible.map(([label]) => label).join(", "));
  }
  Object.assign(entry.node, replacement, {children});
  return entry.node;
}

export function layoutTree(root, {compact = false} = {}) {
  const width = compact ? 24 : 204;
  const columnGap = compact ? 136 : 100;
  const rowGap = compact ? 28 : 24;
  const padding = compact ? 64 : 32;
  const entries = walkTree(root);
  const measured = new Map();
  for (let entryIndex = entries.length - 1; entryIndex >= 0; entryIndex--) {
    const entry = entries[entryIndex];
    let labels;
    try { labels = branchLabels(entry.node); } catch { labels = Object.keys(entry.node.children || {}); }
    const height = compact ? 56 : 78 + labels.length * 30;
    const children = labels.flatMap(label => (entry.node.children?.[label] || []).map(node => ({node, label})));
    const span = Math.max(height, children.reduce((sum, child) => sum + measured.get(child.node).span, 0) + Math.max(0, children.length - 1) * rowGap);
    measured.set(entry.node, {labels, height, children, span});
  }
  const nodes = [];
  const edges = [];
  function place(node, depth, top, parentId = null, branch = null) {
    const measure = measured.get(node);
    const x = padding + depth * (width + columnGap);
    const y = top + (measure.span - measure.height) / 2;
    nodes.push({node, x, y, width, height: measure.height, labels: measure.labels, depth, parentId, branch});
    const childSpan = measure.children.reduce((sum, child) => sum + measured.get(child.node).span, 0) + Math.max(0, measure.children.length - 1) * rowGap;
    let childTop = top + (measure.span - childSpan) / 2;
    for (const child of measure.children) {
      const childMeasure = measured.get(child.node);
      edges.push({parentId: node.id, childId: child.node.id, branch: child.label,
        fromX: x + (compact ? 10 : width), fromY: y + (compact ? 10 : 84 + measure.labels.indexOf(child.label) * 30),
        toX: x + width + columnGap + (compact ? 10 : 0),
        toY: childTop + (childMeasure.span - childMeasure.height) / 2 + (compact ? 10 : 34)});
      place(child.node, depth + 1, childTop, node.id, child.label);
      childTop += childMeasure.span + rowGap;
    }
  }
  place(root, 0, padding);
  return {nodes, edges, width: Math.max(...nodes.map(node => node.x + width)) + padding,
    height: measured.get(root).span + padding * 2, levels: Math.max(...entries.map(entry => entry.depth)) + 1};
}

export async function executeWorkflow(config, evaluate, {signal, onProgress} = {}) {
  const checked = validateCompositionConfig(config);
  const visited = [];
  const decisions = [];
  const pending = [checked.root];
  const throwIfAborted = () => {
    if (signal?.aborted) throw new DOMException("Stopped waiting.", "AbortError");
  };
  while (pending.length) {
    throwIfAborted();
    const node = pending.shift();
    const {id, children, ...question} = node;
    const request = {model: checked.model, state: checked.state, questions: Object.fromEntries([[id, question]])};
    const response = await evaluate(request, signal);
    throwIfAborted();
    const answer = response?.answers?.[id];
    if (!answer || answer.type !== node.type) throw new Error("Missing or invalid answer for " + id + ".");
    let prediction;
    if (node.type === "noul") {
      if (typeof answer.noul !== "number" || !Number.isFinite(answer.noul) || answer.noul < 0 || answer.noul > 1)
        throw new Error("Invalid Noul probability for " + id + ".");
      prediction = answer.noul >= 0.5 ? "true" : "false";
    } else {
      const labels = branchLabels(node);
      const probabilities = answer.probabilities;
      if (!probabilities || labels.some(label => !Object.hasOwn(probabilities, label) ||
          typeof probabilities[label] !== "number" || !Number.isFinite(probabilities[label]) ||
          probabilities[label] < 0 || probabilities[label] > 1) || Object.keys(probabilities).some(label => !labels.includes(label)))
        throw new Error("Invalid decision probabilities for " + id + ".");
      prediction = labels.reduce((best, label) => probabilities[label] > probabilities[best] ? label : best, labels[0]);
    }
    visited.push(id);
    decisions.push({id, type: node.type, prediction, answer: structuredClone(answer), usage: response.usage});
    onProgress?.({id, prediction, completed: decisions.length});
    pending.push(...(children && Object.hasOwn(children, prediction) ? children[prediction] : []));
  }
  return {version: 1, model: checked.model, visited,
    skipped: walkTree(checked.root).map(entry => entry.node.id).filter(id => !visited.includes(id)),
    decisions};
}
