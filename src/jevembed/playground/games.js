const $ = id => document.getElementById(id);
const NAMES = {dino: "DINO JUMP", blocks: "TETRIS", snake: "SNAKE", racing: "RACING"};
const PIECES = {
  O: [[0, 0], [1, 0], [0, 1], [1, 1]],
  I: [[0, 0], [1, 0], [2, 0], [3, 0]],
  T: [[0, 0], [1, 0], [2, 0], [1, 1]],
};

function rect(ctx, x, y, width, height, color = "#f4f4f4") {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, width, height);
}

function drawDino(ctx, state, jumpHeight = 0) {
  rect(ctx, 0, 112, 240, 2);
  for (let x = 4; x < 240; x += 18) rect(ctx, x, 119 + (x % 3), 7, 2, "#6d7377");
  for (const [x, y] of [[24, 29], [112, 20], [188, 39]]) {
    rect(ctx, x, y, 17, 2, "#7b8185"); rect(ctx, x + 4, y - 4, 9, 4, "#7b8185");
  }
  const x = 29, y = 70 - Math.round(jumpHeight);
  if (jumpHeight > 0) rect(ctx, x + 13, 110, 24, 2, "#747a7e");
  // A side-on T-rex: broad jaw, short forearm, heavy hips and a tapered tail.
  rect(ctx, x + 23, y, 18, 3);
  rect(ctx, x + 21, y + 3, 23, 10);
  rect(ctx, x + 21, y + 13, 11, 6);
  rect(ctx, x + 31, y + 15, 10, 3);
  rect(ctx, x + 16, y + 16, 15, 15);
  rect(ctx, x + 12, y + 22, 9, 11);
  rect(ctx, x + 7, y + 22, 8, 7);
  rect(ctx, x + 3, y + 19, 6, 7);
  rect(ctx, x, y + 15, 3, 8);
  rect(ctx, x + 29, y + 21, 8, 3);
  rect(ctx, x + 35, y + 23, 2, 3);
  rect(ctx, x + 15, y + 30, 7, 7);
  rect(ctx, x + 15, y + 37, 3, 5);
  rect(ctx, x + 15, y + 40, 7, 2);
  rect(ctx, x + 25, y + 30, 6, 5);
  rect(ctx, x + 28, y + 35, 3, 7);
  rect(ctx, x + 28, y + 40, 7, 2);
  rect(ctx, x + 26, y + 5, 3, 3, "#0c0e0f");
  const cactusX = Math.min(222, 55 + state.distance * 8);
  rect(ctx, cactusX, 90, 8, 22);
  rect(ctx, cactusX - 5, 96, 5, 4);
  rect(ctx, cactusX - 5, 91, 3, 8);
  rect(ctx, cactusX + 8, 101, 5, 4);
  rect(ctx, cactusX + 11, 95, 3, 9);
  ctx.fillStyle = "#e5e7e9";
  ctx.font = "8px monospace";
  ctx.fillText(`DIST ${state.distance}   SPEED ${state.speed}`, 8, 13);
  ctx.fillText(state.jump_cooldown_turns ? "LANDING COOLDOWN" : "READY TO JUMP", 8, 25);
}

function drawBlocks(ctx, state) {
  const left = 80, top = 13, size = 11;
  rect(ctx, left - 4, top - 4, 74, 118, "#656b70");
  rect(ctx, left - 2, top - 2, 70, 114, "#0d0f10");
  state.board.forEach((row, y) => [...row].forEach((cell, x) => {
    rect(ctx, left + x * size, top + y * size, size - 1, size - 1,
      cell === "#" ? "#f4f4f4" : "#282b2e");
  }));
  ctx.fillStyle = "#e5e7e9";
  ctx.font = "8px monospace";
  ctx.fillText("NEXT", 174, 36);
  for (const [x, y] of PIECES[state.piece]) rect(ctx, 171 + x * 9, 45 + y * 9, 8, 8);
  ctx.fillText(`PIECE ${state.piece}`, 8, 13);
  ctx.fillText("DROP BY COLUMN", 8, 132);
}

function drawSnake(ctx, state) {
  const left = 70, top = 20, size = 12;
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++)
    rect(ctx, left + x * size, top + y * size, size - 1, size - 1, "#222629");
  if (state.food) {
    const [x, y] = state.food;
    rect(ctx, left + x * size + 3, top + y * size + 3, 6, 6, "#f4f4f4");
  }
  state.snake.forEach(([x, y], index) => {
    rect(ctx, left + x * size + 1, top + y * size + 1, 9, 9, index ? "#bfc3c6" : "#fff");
    if (!index) rect(ctx, left + x * size + 6, top + y * size + 3, 2, 2, "#0d0f10");
  });
  ctx.fillStyle = "#e5e7e9";
  ctx.font = "8px monospace";
  ctx.fillText(`HEADING ${state.heading.toUpperCase()}`, 8, 13);
  ctx.fillText("FIND THE FOOD", 8, 132);
}

function drawRacing(ctx, state, motion = null) {
  rect(ctx, 70, 0, 3, 144);
  rect(ctx, 168, 0, 3, 144);
  for (let y = 4; y < 144; y += 18) {
    rect(ctx, 104, y, 2, 10, "#70767b");
    rect(ctx, 136, y, 2, 10, "#70767b");
  }
  const crashed = state.done && state.score < state.turn;
  const rows = motion ? motion.from.hazards : state.hazards;
  const advance = motion ? motion.progress * 29 : crashed ? 29 : 0;
  function barrier(row, y) {
    for (const lane of row) {
      const x = 77 + lane * 32;
      rect(ctx, x, Math.round(y), 23, 13, "#f4f4f4");
      rect(ctx, x + 4, Math.round(y) + 3, 15, 7, "#0d0f10");
    }
  }
  rows.forEach((row, index) => barrier(row, 76 - index * 29 + advance));
  if (motion && !crashed) barrier(state.hazards[2], -11 + advance);
  const lane = motion ? motion.from.lane + (state.lane - motion.from.lane) * motion.progress : state.lane;
  const carX = Math.round(77 + lane * 32), carY = 98;
  // Open wheels and separate wings distinguish the player from road barriers.
  rect(ctx, carX + 2, carY, 19, 3);
  rect(ctx, carX + 10, carY + 3, 3, 10);
  rect(ctx, carX + 4, carY + 9, 15, 2, "#a6adb3");
  rect(ctx, carX + 8, carY + 12, 7, 5);
  rect(ctx, carX + 6, carY + 17, 11, 9);
  rect(ctx, carX + 9, carY + 26, 5, 4);
  for (const wheelX of [carX, carX + 18]) {
    for (const wheelY of [carY + 6, carY + 21]) {
      rect(ctx, wheelX, wheelY, 5, 8, "#a6adb3");
      rect(ctx, wheelX + 1, wheelY + 1, 3, 6, "#25292d");
    }
  }
  rect(ctx, carX + 9, carY + 16, 5, 7, "#0c0e0f");
  rect(ctx, carX + 10, carY + 17, 3, 3, "#a6adb3");
  rect(ctx, carX + 2, carY + 30, 19, 3);
  ctx.fillStyle = "#e5e7e9";
  ctx.font = "8px monospace";
  ctx.fillText(`LANE ${state.lane + 1}`, 8, 13);
  ctx.fillText("AVOID THE NEXT ROW", 8, 142);
}

function draw(canvas, data, jumpHeight = 0, racingMotion = null) {
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  rect(ctx, 0, 0, 240, 144, "#0c0e0f");
  if (!data) return;
  const state = data.state;
  if (state.kind === "dino") drawDino(ctx, state, jumpHeight);
  else if (state.kind === "blocks") drawBlocks(ctx, state);
  else if (state.kind === "snake") drawSnake(ctx, state);
  else drawRacing(ctx, state, racingMotion);
  if (state.done) {
    rect(ctx, 42, 55, 156, 32, "#0c0e0f");
    ctx.strokeStyle = "#f4f4f4";
    ctx.strokeRect(42.5, 55.5, 155, 31);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 10px monospace";
    ctx.fillText(data.event.includes("Game over") ? "GAME OVER" : "RUN COMPLETE", 82, 75);
  }
}

async function jsonResponse(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`);
  return body;
}

export function initGames() {
  const ui = {
    root: $("game-lab"), picker: document.querySelectorAll(".game-picker [data-game]"),
    model: $("game-model"), step: $("game-step"), auto: $("game-auto"),
    reset: $("game-reset"), replay: $("game-replay"), stop: $("game-stop"),
    status: $("game-status"), name: $("game-name"), turn: $("game-turn"),
    canvas: $("game-canvas"), screen: document.querySelector(".game-screen-card"),
    caption: $("game-caption"), score: $("game-score"), scoreLabel: $("game-score-label"),
    count: $("game-count"), lastAction: $("game-last-action"),
    lastEvent: $("game-last-event"), probabilities: $("game-probabilities"),
    input: $("game-input-json"),
  };
  const sessions = new Map();
  const seeds = {dino: 0, blocks: 0, snake: 0, racing: 0};
  let kind = "dino";
  let active = false;
  let running = false;
  let autoplay = false;
  let controller = null;
  let thinkingTimer = null;
  let sequence = 0;

  function current() { return sessions.get(kind) || null; }

  function modelRequest(data, modelId = ui.model.value) {
    return {model: modelId, state: data.model_state, questions: {decision: data.question}};
  }

  function render() {
    const session = current();
    const data = session?.data || null;
    for (const button of ui.picker) button.setAttribute("aria-pressed", String(button.dataset.game === kind));
    ui.name.textContent = NAMES[kind];
    ui.canvas.setAttribute("aria-label", `${NAMES[kind]} game state`);
    ui.turn.textContent = `TURN ${String(data?.stats.turn || 0).padStart(2, "0")} / 12`;
    ui.caption.textContent = data?.event || "Loading a new run…";
    ui.score.textContent = String(data?.stats.score || 0);
    ui.scoreLabel.textContent = data?.stats.score_label || "Score";
    ui.count.textContent = `${data?.stats.turn || 0} / ${data?.stats.limit || 12}`;
    ui.lastAction.textContent = session?.last?.action || "—";
    ui.lastEvent.textContent = session?.last ? `${session.last.event} · ${Math.round(session.last.elapsed)} ms` :
      "Choose a model to begin.";
    ui.probabilities.replaceChildren();
    if (session?.last?.probabilities) {
      for (const [label, value] of Object.entries(session.last.probabilities)) {
        const row = document.createElement("div");
        row.className = "game-probability";
        const title = document.createElement("span");
        title.textContent = label;
        const amount = document.createElement("strong");
        amount.textContent = `${(Number(value) * 100).toFixed(1)}%`;
        row.append(title, amount);
        ui.probabilities.append(row);
      }
    }
    ui.input.textContent = data?.question ? JSON.stringify(modelRequest(data), null, 2) : "Run complete.";
    ui.step.disabled = running || !data?.question || !ui.model.value;
    ui.auto.disabled = running || !data?.question || !ui.model.value;
    ui.auto.textContent = autoplay ? "Playing to the end…" : "Play to the end";
    ui.step.textContent = running ? "Model thinking…" : "Ask model for next move ↗";
    ui.reset.hidden = running;
    ui.stop.hidden = !running;
    ui.replay.disabled = running;
    ui.model.disabled = running;
    const modelTrigger = $("game-model-trigger");
    if (modelTrigger) modelTrigger.disabled = running;
    ui.screen.classList.toggle("is-thinking", running);
    draw(ui.canvas, data);
  }

  function animateJump(data, thisSequence) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return Promise.resolve();
    return new Promise(resolve => {
      const started = performance.now();
      function frame(now) {
        if (thisSequence !== sequence) { resolve(); return; }
        const progress = Math.min(1, (now - started) / 650);
        draw(ui.canvas, data, Math.sin(Math.PI * progress) * 29);
        if (progress < 1) requestAnimationFrame(frame);
        else resolve();
      }
      requestAnimationFrame(frame);
    });
  }

  function animateRacing(from, data, thisSequence) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return Promise.resolve();
    return new Promise(resolve => {
      const started = performance.now();
      function frame(now) {
        if (thisSequence !== sequence) { resolve(); return; }
        const progress = Math.min(1, (now - started) / 550);
        draw(ui.canvas, data, 0, {from, progress});
        if (progress < 1) requestAnimationFrame(frame);
        else resolve();
      }
      requestAnimationFrame(frame);
    });
  }

  function stop() {
    sequence++;
    controller?.abort();
    controller = null;
    clearInterval(thinkingTimer);
    thinkingTimer = null;
    running = false;
    autoplay = false;
  }

  async function newRun(nextKind = kind, seed = seeds[nextKind]) {
    stop();
    kind = nextKind;
    ui.status.textContent = "Loading game…";
    ui.status.classList.remove("is-error");
    render();
    const thisSequence = sequence;
    try {
      const data = await jsonResponse(`/playground/api/games/new?kind=${encodeURIComponent(kind)}&seed=${seed}`);
      if (thisSequence !== sequence) return;
      sessions.set(kind, {data, last: null});
      ui.status.textContent = "Ready for the next decision.";
      render();
    } catch (error) {
      if (thisSequence !== sequence) return;
      ui.status.textContent = error.message;
      ui.status.classList.add("is-error");
    }
  }

  async function modelStep(modelId = ui.model.value) {
    const session = current();
    if (!session?.data?.question || running || !modelId) return false;
    controller = new AbortController();
    const signal = controller.signal;
    const thisSequence = ++sequence;
    running = true;
    ui.status.textContent = "Waiting for the model…";
    ui.status.classList.remove("is-error");
    render();
    const request = modelRequest(session.data, modelId);
    const start = performance.now();
    const timer = setInterval(() => {
      if (thisSequence !== sequence || !running) return;
      const seconds = Math.floor((performance.now() - start) / 1000);
      ui.step.textContent = `Model thinking… ${seconds}s`;
      if (seconds >= 5 && seconds % 5 === 0)
        ui.status.textContent = `Waiting ${seconds}s for ${modelId}. First use may load model weights.`;
    }, 1000);
    thinkingTimer = timer;
    try {
      const response = await jsonResponse("/v1/systemone", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request), signal,
      });
      const elapsed = performance.now() - start;
      const answer = response?.answers?.decision;
      if (answer?.type !== "choice" || !Object.hasOwn(session.data.question.criteria, answer.choice))
        throw new Error("The model returned an invalid game action.");
      const data = await jsonResponse("/playground/api/games/step", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({state: session.data.state, action: answer.choice}), signal,
      });
      if (thisSequence !== sequence) return false;
      sessions.set(kind, {data, last: {action: answer.choice, probabilities: answer.probabilities,
        event: data.event, elapsed}});
      clearInterval(timer);
      render();
      if (data.state.kind === "dino" && data.animation === "jump") {
        ui.step.textContent = "Jumping…";
        ui.status.textContent = "Applying the model's move…";
        await animateJump(data, thisSequence);
        if (thisSequence !== sequence) return false;
      } else if (data.state.kind === "racing") {
        ui.step.textContent = "Moving…";
        ui.status.textContent = "Applying the model's move…";
        await animateRacing(session.data.state, data, thisSequence);
        if (thisSequence !== sequence) return false;
      }
      ui.status.textContent = data.state.done ? "Run finished. Start a new run to try again." : "Move applied.";
      return true;
    } catch (error) {
      if (thisSequence !== sequence) return false;
      ui.status.textContent = error.name === "AbortError" ? "Stopped waiting for the model." : error.message;
      ui.status.classList.add("is-error");
      return false;
    } finally {
      clearInterval(timer);
      if (thinkingTimer === timer) thinkingTimer = null;
      if (thisSequence === sequence) {
        controller = null;
        running = false;
        render();
      }
    }
  }

  async function playToEnd() {
    if (running || !current()?.data?.question || !ui.model.value) return;
    const modelId = ui.model.value;
    autoplay = true;
    render();
    while (active && autoplay && current()?.data?.question) {
      if (!await modelStep(modelId)) break;
    }
    autoplay = false;
    render();
  }

  for (const button of ui.picker) button.addEventListener("click", () => {
    if (button.dataset.game === kind) return;
    stop();
    kind = button.dataset.game;
    ui.status.textContent = "";
    if (!sessions.has(kind)) newRun(kind);
    else render();
  });
  ui.step.addEventListener("click", () => modelStep());
  ui.auto.addEventListener("click", playToEnd);
  ui.stop.addEventListener("click", () => {
    stop();
    ui.status.textContent = "Stopped waiting. The server may still finish the current request.";
    render();
  });
  ui.reset.addEventListener("click", () => {
    seeds[kind] = (seeds[kind] + 1) % 1000000;
    newRun(kind, seeds[kind]);
  });
  ui.replay.addEventListener("click", () => newRun(kind, seeds[kind]));
  ui.model.addEventListener("change", render);
  render();
  return {
    show() { active = true; if (!current()) newRun(); else render(); },
    hide() { active = false; stop(); render(); },
    setModels(ids) {
      const selected = ui.model.value;
      ui.model.replaceChildren(...(ids.length ? ids : [""]).map(id => {
        const entry = document.createElement("option");
        entry.value = id;
        entry.textContent = id || "No models available";
        return entry;
      }));
      if (ids.includes(selected)) ui.model.value = selected;
      render();
    },
  };
}
