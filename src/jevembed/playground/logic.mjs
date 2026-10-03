export const PRESETS = {
  choice: {
    name: "Support routing", type: "choice",
    state: "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
    instructions: "Which team should handle this request?",
    criteria: "returns: Exchanges, wrong or damaged items\nshipping: Delivery status, delays, lost packages\nbilling: Charges, invoices, payment problems",
  },
  score: {
    name: "Issue severity", type: "score",
    state: "The export button crashes the settings page in Safari. It works in Chrome, but several customers only use Safari.",
    instructions: "How severe is the reported issue?",
    criteria: "Cosmetic; no impact to functionality\nBroken or degraded feature, but a workaround exists\nBlocking issue; no workaround exists",
  },
  noul: {
    name: "Human escalation", type: "noul",
    state: "I have asked three times now. Can I please just talk to a real person?",
    instructions: "Is the customer asking for a human agent?",
    criteria: "",
  },
};

const isContent = value => typeof value === "string" ||
  (value !== null && (Array.isArray(value) || typeof value === "object"));

function requireFiniteNumbers(value, name) {
  const pending = [value];
  while (pending.length) {
    const current = pending.pop();
    if (typeof current === "number" && !Number.isFinite(current))
      throw new Error(`${name} contains a number too large for JSON.`);
    if (current !== null && typeof current === "object") {
      for (const child of Object.values(current)) pending.push(child);
    }
  }
}

function parseCriteria(type, source, advanced) {
  if (advanced) {
    let parsed;
    try { parsed = JSON.parse(source); }
    catch { throw new Error("Criteria must be valid JSON."); }
    requireFiniteNumbers(parsed, "Criteria");
    if (type === "choice") {
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object" || !Object.keys(parsed).length)
        throw new Error("Choice criteria must be a nonempty JSON object.");
      if (Object.keys(parsed).some(label => !label.trim()))
        throw new Error("Choice labels must be nonempty, including in JSON mode.");
      if (Object.values(parsed).some(value => value !== null && !isContent(value)))
        throw new Error("Choice descriptions must be text, JSON objects, arrays, or null.");
    } else if (type === "score") {
      if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 10 || parsed.some(value => !isContent(value)))
        throw new Error("Score criteria must be a JSON array of 2 to 10 descriptions.");
    } else if (parsed !== null && (!parsed || Array.isArray(parsed) || typeof parsed !== "object" ||
               Object.keys(parsed).sort().join() !== "false,true" || Object.values(parsed).some(value => !isContent(value)))) {
      throw new Error('Noul criteria must be null or a JSON object with "true" and "false" descriptions.');
    }
    return parsed;
  }
  const lines = source.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if (type === "choice") {
    const criteria = new Map();
    for (const line of lines) {
      const colon = line.indexOf(":");
      const label = (colon < 0 ? line : line.slice(0, colon)).trim();
      if (!label || criteria.has(label)) throw new Error("Choice labels must be unique and nonempty.");
      criteria.set(label, colon < 0 || !line.slice(colon + 1).trim() ? null : line.slice(colon + 1).trim());
    }
    if (!lines.length) throw new Error("Add at least one choice.");
    return Object.fromEntries(criteria);
  }
  if (type === "score") {
    if (lines.length < 2 || lines.length > 10) throw new Error("Add 2 to 10 score levels, one per line.");
    return lines;
  }
  if (lines.length) throw new Error("Noul has no criteria in simple mode. Turn on JSON criteria to add them.");
  return undefined;
}

export function buildQuestion({type, instructions, criteria, advanced = false}) {
  if (!["choice", "score", "noul"].includes(type)) throw new Error("Choose a question type.");
  if (!instructions.trim()) throw new Error("Add a question or instruction.");
  const question = {type, instructions: instructions.trim()};
  const parsed = parseCriteria(type, criteria, advanced);
  if (type !== "noul" || parsed !== undefined && parsed !== null) question.criteria = parsed;
  return question;
}

export function buildRequest({model, state, jsonState, type, instructions, criteria, advanced}) {
  if (!model) throw new Error("Choose a model.");
  let content = state;
  if (jsonState) {
    try { content = JSON.parse(state); }
    catch { throw new Error("State must be valid JSON."); }
    requireFiniteNumbers(content, "State");
    if (!isContent(content)) throw new Error("JSON state must be text, an object, or an array.");
  }
  if (typeof content === "string" && !content.trim()) throw new Error("Add a state to evaluate.");
  return {model, state: content, questions: {decision: buildQuestion({type, instructions, criteria, advanced})}};
}

export const COMPOSITION_LIMITS = {maxDepth: 128, maxNodes: 2048};

function parseCompositionState(state, jsonState) {
  let content = state;
  if (jsonState) {
    try { content = JSON.parse(state); }
    catch { throw new Error("State must be valid JSON."); }
  }
  if (!isContent(content)) throw new Error("State must be text, an object, or an array.");
  if (typeof content === "string" && !content.trim()) throw new Error("Add a state to evaluate.");
  requireFiniteNumbers(content, "State");
  return content;
}

function normalizeNodeId(value, fallback) {
  const id = value === undefined && fallback ? fallback : value;
  if (typeof id !== "string" || !/^[A-Za-z][A-Za-z0-9_-]*$/.test(id))
    throw new Error("Node IDs must start with a letter and use letters, numbers, _ or -.");
  return id;
}

function compositionQuestion(node, drafts) {
  if (drafts && typeof node.criteria === "string") {
    let instructions = node.instructions;
    if (node.instructionsAdvanced) {
      try { instructions = JSON.parse(instructions); }
      catch { throw new Error("Instructions must be valid JSON."); }
    }
    if (!isContent(instructions) || typeof instructions === "string" && !instructions.trim())
      throw new Error("Add a question or instruction.");
    const question = buildQuestion({...node,
      instructions: typeof instructions === "string" ? instructions : "Structured instructions"});
    question.instructions = structuredClone(instructions);
    requireFiniteNumbers(question, "Question");
    return question;
  }
  if (!["choice", "score", "noul"].includes(node.type)) throw new Error("Choose a question type.");
  if (!isContent(node.instructions) || typeof node.instructions === "string" && !node.instructions.trim())
    throw new Error("Add a question or instruction.");
  const question = {type: node.type, instructions: structuredClone(node.instructions)};
  if (node.type === "choice") {
    if (!node.criteria || Array.isArray(node.criteria) || typeof node.criteria !== "object" ||
        !Object.keys(node.criteria).length || Object.keys(node.criteria).some(label => !label.trim()) ||
        Object.values(node.criteria).some(value => value !== null && !isContent(value)))
      throw new Error("Choice criteria must be a nonempty object of named descriptions.");
    question.criteria = structuredClone(node.criteria);
  } else if (node.type === "score") {
    if (!Array.isArray(node.criteria) || node.criteria.length < 2 || node.criteria.length > 10 ||
        node.criteria.some(value => !isContent(value)))
      throw new Error("Score criteria must contain 2 to 10 ordered descriptions.");
    question.criteria = structuredClone(node.criteria);
  } else if (Object.hasOwn(node, "criteria")) {
    if (!node.criteria || Array.isArray(node.criteria) || typeof node.criteria !== "object" ||
        Object.keys(node.criteria).sort().join() !== "false,true" ||
        Object.values(node.criteria).some(value => !isContent(value)))
      throw new Error("Omit Noul criteria or supply both true and false descriptions.");
    question.criteria = structuredClone(node.criteria);
  }
  requireFiniteNumbers(question, "Question");
  return question;
}

function serializeCompositionNode(root, drafts) {
  const seenIds = new Set();
  const seenNodes = new WeakSet();
  let count = 0;
  function visit(node, depth) {
    if (!node || typeof node !== "object" || Array.isArray(node))
      throw new Error("Every workflow node must be an object.");
    if (seenNodes.has(node)) throw new Error("Workflow nodes must form a tree without cycles or shared nodes.");
    seenNodes.add(node);
    if (depth >= COMPOSITION_LIMITS.maxDepth)
      throw new Error("Workflow nesting is limited to " + COMPOSITION_LIMITS.maxDepth + " levels.");
    count += 1;
    if (count > COMPOSITION_LIMITS.maxNodes)
      throw new Error("Workflow cannot contain more than " + COMPOSITION_LIMITS.maxNodes + " nodes.");
    if (!drafts && Object.keys(node).some(key => !["id", "type", "instructions", "criteria", "children"].includes(key)))
      throw new Error("Workflow node contains unsupported fields.");
    const id = normalizeNodeId(node.id, drafts ? "decision_" + count : undefined);
    if (seenIds.has(id)) throw new Error('Node ID "' + id + '" must be unique.');
    seenIds.add(id);
    const question = compositionQuestion(node, drafts);
    const labels = labelsForQuestion(question);
    const children = node.children === undefined ? {} : node.children;
    if (!children || typeof children !== "object" || Array.isArray(children))
      throw new Error('Children for "' + id + '" must be an object keyed by prediction label.');
    const branches = [];
    for (const [label, nodes] of Object.entries(children)) {
      if (!labels.includes(label)) throw new Error('Node "' + id + '" has an invalid branch "' + label + '".');
      if (!Array.isArray(nodes)) throw new Error('Branch "' + label + '" must contain an array of nodes.');
      if (nodes.length) branches.push([label, nodes.map(child => visit(child, depth + 1))]);
    }
    const result = {id, ...question};
    if (branches.length) result.children = Object.fromEntries(branches);
    return result;
  }
  return visit(root, 0);
}

export function buildCompositionConfig({model, state, jsonState, root}) {
  if (typeof model !== "string" || !model.trim()) throw new Error("Choose a model.");
  return {version: 1, model, state: parseCompositionState(state, jsonState),
    root: serializeCompositionNode(root, true)};
}

export function validateCompositionConfig(config) {
  if (!config || typeof config !== "object" || Array.isArray(config) ||
      Object.keys(config).some(key => !["version", "model", "state", "root"].includes(key)))
    throw new Error("A workflow needs only version, model, state, and root.");
  if (config.version !== 1) throw new Error("Unsupported workflow version. Expected version 1.");
  if (typeof config.model !== "string" || !config.model.trim()) throw new Error("Choose a model.");
  try { JSON.stringify(config); }
  catch { throw new Error("Workflow must be JSON without circular references."); }
  return {version: 1, model: config.model,
    state: structuredClone(parseCompositionState(config.state, false)),
    root: serializeCompositionNode(config.root, false)};
}

export function flattenCompositionConfig(config) {
  const checked = validateCompositionConfig(config);
  const entries = [];
  function visit(node) {
    const {id, children, ...question} = node;
    entries.push([id, question]);
    for (const nodes of Object.values(children || {})) nodes.forEach(visit);
  }
  visit(checked.root);
  return {model: checked.model, state: checked.state, questions: Object.fromEntries(entries)};
}

export function buildCompositionRequest({model, state, jsonState, steps, root}) {
  if (root) return flattenCompositionConfig(buildCompositionConfig({model, state, jsonState, root}));
  if (typeof model !== "string" || !model.trim()) throw new Error("Choose a model.");
  const content = parseCompositionState(state, jsonState);
  if (!Array.isArray(steps) || !steps.length) throw new Error("Add at least one decision step.");
  const ids = new Set();
  const questions = steps.map((step, index) => {
    const id = normalizeNodeId(step.id, "decision_" + (index + 1));
    if (ids.has(id)) throw new Error('Step ID "' + id + '" must be unique.');
    ids.add(id);
    return [id, buildQuestion(step)];
  });
  return {model, state: content, questions: Object.fromEntries(questions)};
}

export const COMPOSITION_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  title: "JevEmbed recursive decision workflow",
  type: "object",
  additionalProperties: false,
  required: ["version", "model", "state", "root"],
  properties: {
    version: {const: 1}, model: {type: "string", minLength: 1},
    state: {$ref: "#/$defs/content"}, root: {$ref: "#/$defs/node"}
  },
  $defs: {
    content: {type: ["string", "object", "array"]},
    choiceDescription: {type: ["string", "object", "array", "null"]},
    children: {type: "object", additionalProperties: {type: "array", items: {$ref: "#/$defs/node"}}},
    node: {
      type: "object", additionalProperties: false, required: ["id", "type", "instructions"],
      properties: {
        id: {type: "string", pattern: "^[A-Za-z][A-Za-z0-9_-]*$"},
        type: {enum: ["choice", "score", "noul"]},
        instructions: {$ref: "#/$defs/content"},
        criteria: {},
        children: {$ref: "#/$defs/children"}
      },
      allOf: [
        {if: {properties: {type: {const: "choice"}}}, then: {
          required: ["criteria"], properties: {criteria: {type: "object", minProperties: 1,
            propertyNames: {pattern: "\\S"}, additionalProperties: {$ref: "#/$defs/choiceDescription"}}}
        }},
        {if: {properties: {type: {const: "score"}}}, then: {
          required: ["criteria"], properties: {criteria: {type: "array", minItems: 2, maxItems: 10,
            items: {$ref: "#/$defs/content"}}, children: {propertyNames: {pattern: "^[0-9]$"}}}
        }},
        {if: {properties: {type: {const: "noul"}}}, then: {
          properties: {
            criteria: {type: "object", additionalProperties: false, required: ["true", "false"],
              properties: {true: {$ref: "#/$defs/content"}, false: {$ref: "#/$defs/content"}}},
            children: {propertyNames: {enum: ["true", "false"]}}
          }
        }}
      ]
    }
  }
};

export function preferredModelId(models) {
  const entries = Array.isArray(models) ? models.filter(entry => entry && typeof entry.id === "string" && entry.id) : [];
  return entries.find(entry => /jevembed/i.test(entry.id))?.id || entries[0]?.id || "";
}

export function labelsForQuestion(question) {
  if (question.type === "choice") return Object.keys(question.criteria);
  if (question.type === "score") return question.criteria.map((_, index) => String(index));
  return ["true", "false"];
}

export function simpleCriteriaText(question) {
  if (question.type === "choice") {
    if (Object.entries(question.criteria).some(([label, value]) =>
      label !== label.trim() || /[:\r\n]/.test(label) ||
      value !== null && (typeof value !== "string" || !value || value !== value.trim() || /[\r\n]/.test(value))))
      throw new Error("These labels or descriptions need JSON mode to preserve them exactly.");
    return Object.entries(question.criteria).map(([label, value]) => value === null ? label : `${label}: ${value}`).join("\n");
  }
  if (question.type === "score") {
    if (question.criteria.some(value => typeof value !== "string" || !value || value !== value.trim() || /[\r\n]/.test(value)))
      throw new Error("These levels need JSON mode to preserve them exactly.");
    return question.criteria.join("\n");
  }
  if (question.criteria) throw new Error("Noul criteria need JSON mode.");
  return "";
}

export function scoreSummary(answer, question) {
  const score = Number(answer.score);
  return {value: `${Number.isFinite(score) ? score.toFixed(2) : "—"} / ${question.criteria.length - 1}`,
    mostLikely: `Level ${hardPrediction(answer)}`};
}

export function hardPrediction(answer) {
  if (!answer) return null;
  if (answer.type === "noul") return Number(answer.noul) >= 0.5 ? "true" : "false";
  const entries = Object.entries(answer.probabilities || {});
  if (!entries.length) return null;
  return entries.reduce((best, entry) => Number(entry[1]) > Number(best[1]) ? entry : best)[0];
}

export function comparison(previous, current) {
  if (!previous || !current || previous.request.model !== current.request.model ||
      JSON.stringify(previous.request.state) !== JSON.stringify(current.request.state) ||
      previous.request.questions.decision.type !== current.request.questions.decision.type) return null;
  const before = hardPrediction(previous.answer), after = hardPrediction(current.answer);
  const chance = (answer, label) => answer.type === "noul" ?
    (label === "true" ? Number(answer.noul) : 1 - Number(answer.noul)) : Number(answer.probabilities?.[label]);
  const delta = chance(current.answer, after) - chance(previous.answer, after);
  return {before, after, changed: before !== after, delta};
}

export function trainingRecord(run, correctedLabel, id = "playground-example", secondRun = null) {
  if (!run || correctedLabel === "") throw new Error("Choose a correct answer before exporting.");
  const question = run.request.questions.decision;
  if (!labelsForQuestion(question).includes(correctedLabel)) throw new Error("The corrected answer is not valid for this run.");
  const answer = question.type === "choice" ? {choice: correctedLabel} :
    question.type === "score" ? {level: Number(correctedLabel)} : {noul: correctedLabel === "true"};
  const record = {id, request: {state: run.request.state, questions: {decision: question}}, answers: {decision: answer}};
  const predictions = [run, secondRun].filter(Boolean).flatMap(result => {
    const prediction = result.response?.answers?.decision ?? result.answer;
    return prediction ? [{model: result.request.model, prediction: structuredClone(prediction)}] : [];
  });
  if (predictions.length) record.metadata = {predictions};
  return record;
}
