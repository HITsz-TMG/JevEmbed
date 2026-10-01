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

export function buildCompositionRequest({model, state, jsonState, steps}) {
  if (!model) throw new Error("Choose a model.");
  let content = state;
  if (jsonState) {
    try { content = JSON.parse(state); }
    catch { throw new Error("State must be valid JSON."); }
    requireFiniteNumbers(content, "State");
    if (!isContent(content)) throw new Error("JSON state must be text, an object, or an array.");
  }
  if (typeof content === "string" && !content.trim()) throw new Error("Add a state to evaluate.");
  if (!Array.isArray(steps) || !steps.length) throw new Error("Add at least one decision step.");
  const questions = {};
  steps.forEach((step, index) => {
    const id = String(step.id || `decision_${index + 1}`).trim();
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(id)) throw new Error("Step IDs must start with a letter and use letters, numbers, _ or -.");
    if (questions[id]) throw new Error(`Step ID \"${id}\" must be unique.`);
    questions[id] = buildQuestion(step);
  });
  return {model, state: content, questions};
}

export const COMPOSITION_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  title: "JevEmbed decision composition",
  type: "object",
  additionalProperties: false,
  required: ["model", "state", "questions"],
  properties: {
    model: {type: "string", minLength: 1},
    state: {oneOf: [{type: "string"}, {type: "object"}, {type: "array"}]},
    questions: {
      type: "object", minProperties: 1,
      additionalProperties: {oneOf: [
        {type: "object", additionalProperties: false, required: ["type", "instructions", "criteria"], properties: {
          type: {const: "choice"}, instructions: {type: "string", minLength: 1},
          criteria: {type: "object", minProperties: 1, additionalProperties: {oneOf: [{type: "string"}, {type: "object"}, {type: "array"}, {type: "null"}]}}
        }},
        {type: "object", additionalProperties: false, required: ["type", "instructions", "criteria"], properties: {
          type: {const: "score"}, instructions: {type: "string", minLength: 1},
          criteria: {type: "array", minItems: 2, maxItems: 10, items: {oneOf: [{type: "string"}, {type: "object"}, {type: "array"}, {type: "null"}]}}
        }},
        {type: "object", additionalProperties: false, required: ["type", "instructions"], properties: {
          type: {const: "noul"}, instructions: {type: "string", minLength: 1},
          criteria: {type: "object", required: ["true", "false"], additionalProperties: false}
        }}
      ]}
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
