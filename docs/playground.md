# Playground

[Back to README](../README.md#http-interfaces)

The Playground is a small browser interface for trying Jev Choice, Score, or Noul decisions. It calls the same `/v1/models` and `/v1/systemone` endpoints as other HTTP clients.

From the project root, install local inference and server dependencies with the [tested model constraints](../README.md#installation):

```bash
python -m pip install -e '.[local,server]' -c requirements-models-tested.txt
python -m jevembed --playground
```

By default, the Playground registers [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [JevEmbed-Qwen3-Embedding-0.6B](https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B) using their Hugging Face model IDs. This command also works outside the source directory after installation. To choose your own models, provide `--config`; explicit configurations replace the defaults:

```bash
python -m jevembed \
  --config configs/qwen3-embedding-0.6b.yaml \
  --config configs/jevembed-qwen3-embedding-0.6b.yaml \
  --playground
```

Open <http://127.0.0.1:8000/playground/>. `--playground` starts the HTTP server; no JavaScript build is needed. The server binds to `127.0.0.1` by default. Use `--host` and `--port` if needed. Model weights load on the first run for each model and stay resident, so loading both models needs enough memory for both. A first run may also download weights that are not cached locally.

Use **Playground** for one model, or open **Compare models** in the left sidebar to evaluate the same input with two different registered models. The comparison page is also available at `/playground/#/compare` and requires at least two registered models. Switching pages keeps each page's draft and completed results until you reload.

Use **Compose workflow** at `/playground/#/compose` to combine multiple Jev primitives into one request. The editor uses one shared state and lets you add, remove, reorder, and configure Choice, Score, and Noul steps. Each step needs a unique question ID; the preview shows the exact multi-question JSON sent to `/v1/systemone`. **Copy**, **Download**, and **Download JSON Schema** make the configuration reusable from application code. The server evaluates all questions in one request and returns an answer under each step ID.

**Game Lab** at `/playground/#/games` lets a model play four small, monochrome pixel games: Dino jump, Tetris-style falling blocks, Snake, and Racing. Choose a game and model, then step through moves or select **Play to the end** for a complete run. Every move comes from the selected embedding model through the same Choice API; the simulator supplies observations and applies the returned action. The board, move, probabilities, latency, and game score update together. Use **Replay the same start** to try another model on identical initial conditions; **New run** changes the starting seed. Runs stop after at most 12 turns. Falling blocks use fixed piece orientations. Tetris-style blocks and Snake show game outcomes rather than an artificial per-move accuracy label.

The simulator also describes immediate observations: Dino reports whether a cactus arrives this turn and whether jumping is available; Snake describes each candidate's adjacent cell and whether it moves closer to food. These descriptions use the game state, without choosing or replacing the model's action. Dino has a one-turn landing cooldown after each jump. Open **See what the model reads** to inspect the exact input.

Choose a preset, edit the state, instructions, and criteria, and run the example. State can be plain text or JSON. Results show each model's prediction, probabilities, and elapsed browser time. Switch between Visual and JSON without rerunning inference; JSON shows the complete original API response, including `model`, `answers`, and `usage`. The previous-run indication helps spot a change after editing a rule; it does not measure quality.

If you know the expected answer, correct the label and export that **one labeled example** as JSONL for the [training supervision format](training.md#supervision-format-and-objectives). The corrected label is the training `answers` field; `metadata.predictions` keeps the original response for each model that finished (one in Playground, up to two in Compare). The trainer ignores metadata. A model comparison is a demonstration on one case, not dataset accuracy. The elapsed time includes browser and network overhead and may include model loading and cache effects; the two models run sequentially. It is not a benchmark.

The Playground uses the server's existing [request limits and responses](http.md#limits-and-responses), including 413 for oversized work. Requests sent from the browser are handled by the selected server. Keep the default local bind address for a local demo; if you expose the server on a network, treat submitted states as data sent to that server.
