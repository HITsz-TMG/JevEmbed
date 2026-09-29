import test from "node:test";
import assert from "node:assert/strict";
import {buildQuestion, buildCompositionRequest, buildRequest, comparison, hardPrediction, scoreSummary, simpleCriteriaText, trainingRecord} from "../../src/jevembed/playground/logic.mjs";

const draft = {model: "base", state: "A customer asks for a person", jsonState: false,
  type: "choice", instructions: "Route this", criteria: "human: Needs an agent\nbot: Can self-serve", advanced: false};

test("builds a simple Choice request and validates labels", () => {
  const request = buildRequest(draft);
  assert.deepEqual(request.questions.decision.criteria, {human: "Needs an agent", bot: "Can self-serve"});
  assert.throws(() => buildRequest({...draft, criteria: "human: One\nhuman: Two"}), /unique/);
});

test("Choice labels can omit descriptions without changing their labels", () => {
  const question = buildQuestion({type: "choice", instructions: "Route", criteria: "human\nbot: Self service\nother:"});
  assert.deepEqual(question.criteria, {human: null, bot: "Self service", other: null});
  assert.equal(simpleCriteriaText(question), "human\nbot: Self service\nother");
});

test("JSON to simple mode rejects criteria that would change meaning", () => {
  assert.throws(() => simpleCriteriaText({type: "choice", criteria: {"a:b": "text"}}), /preserve/);
  assert.throws(() => simpleCriteriaText({type: "choice", criteria: {a: "first\nsecond"}}), /preserve/);
  assert.throws(() => simpleCriteriaText({type: "score", criteria: ["low", "high\nimpact"]}), /preserve/);
});

test("accepts JSON state and rich criteria", () => {
  const request = buildRequest({...draft, jsonState: true, state: '{"turn":2}',
    advanced: true, criteria: '{"human":{"reason":"escalation"},"bot":null}'});
  assert.deepEqual(request.state, {turn: 2});
  assert.deepEqual(request.questions.decision.criteria.human, {reason: "escalation"});
  assert.throws(() => buildRequest({...draft, jsonState: true, state: "{"}), /valid JSON/);
});

test("advanced Choice rejects empty labels reserved by the correction control", () => {
  assert.throws(() => buildRequest({...draft, advanced: true, criteria: '{"":"empty","human":"agent"}'}), /nonempty/);
  assert.throws(() => buildRequest({...draft, advanced: true, criteria: '{"  ":"empty"}'}), /nonempty/);
});

test("JSON state and nested criteria reject nonfinite numeric values", () => {
  assert.throws(() => buildRequest({...draft, jsonState: true, state: '{"nested":[1e999]}'}), /too large/);
  assert.throws(() => buildRequest({...draft, advanced: true,
    criteria: '{"human":{"weight":[1e999]},"bot":null}'}), /too large/);
});

test("Score hard prediction uses argmax, Noul uses 0.5 threshold", () => {
  const score = {type: "score", score: 0.9, probabilities: {"0": .4, "1": .35, "2": .25}};
  assert.equal(hardPrediction(score), "0");
  assert.deepEqual(scoreSummary(score, {type: "score", criteria: ["low", "medium", "high"]}),
    {value: "0.90 / 2", mostLikely: "Level 0"});
  assert.equal(hardPrediction({type: "noul", noul: .5}), "true");
  assert.equal(hardPrediction({type: "noul", noul: .49}), "false");
});

test("comparison requires identical state, type, and model", () => {
  const request = buildRequest(draft);
  const before = {request, answer: {type: "choice", probabilities: {human: .8, bot: .2}}};
  const after = {request: {...request, questions: {decision: {...request.questions.decision, instructions: "New rule"}}},
    answer: {type: "choice", probabilities: {human: .3, bot: .7}}};
  const change = comparison(before, after);
  assert.deepEqual({before: change.before, after: change.after, changed: change.changed},
    {before: "human", after: "bot", changed: true});
  assert.ok(Math.abs(change.delta - .5) < 1e-12);
  assert.equal(comparison(before, {...after, request: {...after.request, state: "Different"}}), null);
});

test("export keeps corrected Choice supervision separate from raw model predictions", () => {
  const request = buildRequest(draft);
  const prediction = {type: "choice", choice: "bot", probabilities: {human: .123456789012345, bot: .876543210987655}};
  const run = {request, answer: prediction, response: {answers: {decision: prediction}}};
  assert.deepEqual(trainingRecord(run, "human", "example-1"), {id: "example-1",
    request: {state: draft.state, questions: request.questions}, answers: {decision: {choice: "human"}},
    metadata: {predictions: [{model: "base", prediction}]}});
  assert.throws(() => trainingRecord(run, "other"), /not valid/);
  const secondPrediction = {type: "choice", choice: "human", probabilities: {human: .7, bot: .3}};
  const second = {request: {...request, model: "other"}, answer: secondPrediction};
  const compared = trainingRecord(run, "human", "example-2", second);
  assert.deepEqual(compared.metadata.predictions, [{model: "base", prediction},
    {model: "other", prediction: secondPrediction}]);
  assert.deepEqual(compared.answers, {decision: {choice: "human"}});
  assert.equal(trainingRecord(run, "human", "example-3", null).metadata.predictions.length, 1);
});

test("Score and Noul exports preserve original numeric predictions", () => {
  const scoreRequest = {model: "score-model", state: "x", questions: {decision:
    buildQuestion({type: "score", instructions: "Rate", criteria: "Low\nHigh"})}};
  const scorePrediction = {type: "score", score: .73456789012345,
    probabilities: {"0": .26543210987655, "1": .73456789012345}};
  const score = {request: scoreRequest, answer: scorePrediction};
  const scoreRecord = trainingRecord(score, "0");
  assert.deepEqual(scoreRecord.answers.decision, {level: 0});
  assert.deepEqual(scoreRecord.metadata.predictions, [{model: "score-model", prediction: scorePrediction}]);

  const noulRequest = {model: "noul-model", state: "x", questions: {decision:
    buildQuestion({type: "noul", instructions: "Ask?", criteria: ""})}};
  const noulPrediction = {type: "noul", noul: .3141592653589793};
  const noulRecord = trainingRecord({request: noulRequest, answer: noulPrediction}, "true");
  assert.deepEqual(noulRecord.answers.decision, {noul: true});
  assert.deepEqual(noulRecord.metadata.predictions, [{model: "noul-model", prediction: noulPrediction}]);
});

test("builds a mixed Choice, Score, and Noul composition for one shared state", () => {
  const request = buildCompositionRequest({model: "base", state: "A failed export needs review", jsonState: false, steps: [
    {id: "route", type: "choice", instructions: "Which team?", criteria: "support\nengineering", advanced: false},
    {id: "severity", type: "score", instructions: "How severe?", criteria: "low\nmedium\nhigh", advanced: false},
    {id: "escalate", type: "noul", instructions: "Needs a human?", criteria: "", advanced: false},
  ]});
  assert.deepEqual(Object.keys(request.questions), ["route", "severity", "escalate"]);
  assert.equal(request.questions.route.type, "choice");
  assert.equal(request.questions.severity.type, "score");
  assert.equal(request.questions.escalate.type, "noul");
});

test("composition validates unique step IDs and JSON shared state", () => {
  assert.throws(() => buildCompositionRequest({model: "base", state: "{}", jsonState: true, steps: [
    {id: "same", type: "noul", instructions: "One", criteria: "", advanced: false},
    {id: "same", type: "noul", instructions: "Two", criteria: "", advanced: false},
  ]}), /unique/);
  const request = buildCompositionRequest({model: "base", state: '{"ticket":7}', jsonState: true, steps: [
    {id: "triage", type: "noul", instructions: "Escalate?", criteria: "", advanced: false},
  ]});
  assert.deepEqual(request.state, {ticket: 7});
});
