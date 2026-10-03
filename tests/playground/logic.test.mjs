import test from "node:test";
import assert from "node:assert/strict";
import {buildQuestion, buildCompositionConfig, buildCompositionRequest, flattenCompositionConfig, buildRequest, comparison, hardPrediction, preferredModelId, scoreSummary, simpleCriteriaText, trainingRecord} from "../../src/jevembed/playground/logic.mjs";
import {executeWorkflow, layoutTree, moveNode, replaceNode, walkTree} from "../../src/jevembed/playground/composition.mjs";

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

test("prefers the JevEmbed checkpoint when the playground has multiple models", () => {
  assert.equal(preferredModelId([{id: "qwen3-embedding-0.6b"}, {id: "jevembed-qwen3-embedding-0.6b"}]), "jevembed-qwen3-embedding-0.6b");
  assert.equal(preferredModelId([{id: "base"}]), "base");
  assert.equal(preferredModelId([]), "");
});

test("builds and flattens a recursive composition tree", () => {
  const config = buildCompositionConfig({model: "base", state: "ticket", jsonState: false, root: {
    id: "route", type: "choice", instructions: "Route", criteria: "support\nengineering", advanced: false,
    children: {support: [{id: "severity", type: "score", instructions: "Rate", criteria: "low\nhigh", advanced: false,
      children: {"1": [{id: "escalate", type: "noul", instructions: "Human?", criteria: "", advanced: false}]}}]}
  }});
  assert.equal(config.version, 1);
  assert.equal(config.root.children.support[0].children["1"][0].type, "noul");
  assert.deepEqual(Object.keys(flattenCompositionConfig(config).questions), ["route", "severity", "escalate"]);
  assert.throws(() => buildCompositionConfig({model: "base", state: "x", root: {
    id: "route", type: "choice", instructions: "Route", criteria: "a\nb", children: {unknown: []}
  }}), /invalid branch/);
});

test("moves a nested subtree between valid result branches and rejects descendants", () => {
  const root = {
    id: "route", type: "choice", instructions: "Route", criteria: "a\nb",
    children: {
      a: [{id: "score", type: "score", instructions: "Rate", criteria: "low\nhigh",
        children: {"1": [{id: "leaf", type: "noul", instructions: "Escalate?", criteria: ""}]} }],
      b: []
    }
  };
  assert.equal(moveNode(root, "score", "route", "b").id, "score");
  assert.equal((root.children.a || []).length, 0);
  assert.equal(root.children.b[0].children["1"][0].id, "leaf");
  assert.throws(() => moveNode(root, "score", "leaf", "true"), /cannot move/);
  assert.equal(moveNode(root, "leaf", "route", "a").id, "leaf");
  assert.throws(() => replaceNode(root, "route", {...root, id: "score"}), /unique/);
});

test("changing a decision type remaps all populated branches in order", () => {
  const root = {id: "route", type: "choice", instructions: "Route", criteria: "a\nb",
    children: {a: [{id: "first", type: "noul", instructions: "First?", criteria: ""}],
      b: [{id: "second", type: "noul", instructions: "Second?", criteria: ""}]}};
  replaceNode(root, "route", {id: "route", type: "noul", instructions: "Escalate?", children: {}});
  assert.equal(root.children.true[0].id, "first");
  assert.equal(root.children.false[0].id, "second");
});

test("recursive configuration preserves structured instructions", () => {
  const config = buildCompositionConfig({model: "base", state: "ticket", root: {
    id: "route", type: "choice", instructions: '{"context":"route","rules":["urgent"]}',
    instructionsAdvanced: true, criteria: "support\nother"
  }});
  assert.deepEqual(config.root.instructions, {context: "route", rules: ["urgent"]});
});

test("lays out every nested node and edge without collapsing levels", () => {
  const root = {id: "root", type: "choice", instructions: "Route", criteria: "a\nb",
    children: {a: [{id: "child", type: "score", instructions: "Rate", criteria: "low\nhigh",
      children: {"1": [{id: "deep", type: "noul", instructions: "Escalate?", criteria: ""}]}}], b: []}};
  const tree = layoutTree(root);
  assert.equal(tree.nodes.length, 3);
  assert.equal(tree.edges.length, 2);
  assert.ok(tree.nodes.find(node => node.node.id === "deep").x > tree.nodes.find(node => node.node.id === "child").x);
  assert.ok(tree.height > 0 && tree.width > 0);
  assert.deepEqual(walkTree(root).map(entry => entry.node.id), ["root", "child", "deep"]);
});

test("executes only the selected recursive path and reports skipped branches", async () => {
  const config = buildCompositionConfig({model: "base", state: "ticket", root: {
    id: "route", type: "choice", instructions: "Route", criteria: "a\nb",
    children: {a: [{id: "follow", type: "noul", instructions: "Person?", criteria: "",
      children: {true: [{id: "handoff", type: "choice", instructions: "Where?", criteria: "x\ny"}]}}], b: [
      {id: "unvisited", type: "score", instructions: "Score", criteria: "low\nhigh"}]}
  }});
  const requests = [];
  const result = await executeWorkflow(config, async request => {
    requests.push(request);
    const id = Object.keys(request.questions)[0];
    if (id === "route") return {answers: {route: {type: "choice", probabilities: {a: 0.8, b: 0.2}}}};
    if (id === "follow") return {answers: {follow: {type: "noul", noul: 0.9}}};
    return {answers: {handoff: {type: "choice", probabilities: {x: 0.25, y: 0.75}}}};
  });
  assert.deepEqual(result.visited, ["route", "follow", "handoff"]);
  assert.deepEqual(result.skipped, ["unvisited"]);
  assert.deepEqual(requests.map(request => Object.keys(request.questions)), [["route"], ["follow"], ["handoff"]]);
});
