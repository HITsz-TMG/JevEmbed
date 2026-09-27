"""Small, deterministic, turn-based games for the optional browser demo."""

from copy import deepcopy

from .errors import ValidationError


KINDS = ("dino", "blocks", "snake", "racing")
MAX_TURNS = 12
BOARD_WIDTH = 6
BOARD_HEIGHT = 10
SNAKE_SIZE = 8
PIECES = {
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
}
DIRECTIONS = {"Up": (0, -1), "Right": (1, 0), "Down": (0, 1), "Left": (-1, 0)}
OPPOSITES = {"Up": "Down", "Right": "Left", "Down": "Up", "Left": "Right"}


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _next(seed, modulus):
    seed = (seed * 1103515245 + 12345) % (2**31)
    return seed, seed % modulus


def _food(seed, snake):
    free = [[x, y] for y in range(SNAKE_SIZE) for x in range(SNAKE_SIZE) if [x, y] not in snake]
    if not free:
        return seed, None
    seed, index = _next(seed, len(free))
    return seed, free[index]


def new_game(kind, seed=0):
    if kind not in KINDS:
        raise ValidationError("Unknown game")
    seed = _integer(seed, "seed", 0, 999999)
    common = {"kind": kind, "seed": seed, "turn": 0, "done": False, "score": 0}
    if kind == "dino":
        seed, distance = _next(seed, 13)
        seed, speed = _next(seed, 3)
        state = {**common, "seed": seed, "distance": distance + 6, "speed": speed + 4,
                 "jump_cooldown_turns": 0}
    elif kind == "blocks":
        state = {**common, "board": ["......"] * 8 + ["##..##", "##..##"], "piece": "O"}
    elif kind == "snake":
        snake = [[3, 4], [2, 4], [1, 4]]
        seed, food = _food(seed, snake)
        state = {**common, "seed": seed, "snake": snake, "food": food, "heading": "Right"}
    else:
        hazards = []
        for _ in range(3):
            seed, lane = _next(seed, 3)
            hazards.append([lane])
        state = {**common, "seed": seed, "lane": 1, "hazards": hazards}
    return snapshot(state, "New run ready.")


def _validated_state(raw):
    if not isinstance(raw, dict) or raw.get("kind") not in KINDS:
        raise ValidationError("Invalid game state")
    kind = raw["kind"]
    common = {"kind", "seed", "turn", "done", "score"}
    fields = {
        "dino": {"distance", "speed", "jump_cooldown_turns"},
        "blocks": {"board", "piece"},
        "snake": {"snake", "food", "heading"},
        "racing": {"lane", "hazards"},
    }[kind]
    if set(raw) != common | fields:
        raise ValidationError("Invalid game state fields")
    _integer(raw["seed"], "seed", 0, 2**31 - 1)
    _integer(raw["turn"], "turn", 0, MAX_TURNS)
    if type(raw["done"]) is not bool:
        raise ValidationError("done must be a boolean")
    _integer(raw["score"], "score", 0, 4 * MAX_TURNS if kind == "blocks" else MAX_TURNS)
    if kind == "dino":
        _integer(raw["distance"], "distance", 1, 24)
        _integer(raw["speed"], "speed", 4, 6)
        _integer(raw["jump_cooldown_turns"], "jump_cooldown_turns", 0, 1)
    elif kind == "blocks":
        board = raw["board"]
        if (not isinstance(board, list) or len(board) != BOARD_HEIGHT or
                any(not isinstance(row, str) or len(row) != BOARD_WIDTH or set(row) - {".", "#"}
                    for row in board) or not isinstance(raw["piece"], str) or raw["piece"] not in PIECES):
            raise ValidationError("Invalid falling-block board or piece")
    elif kind == "snake":
        snake = raw["snake"]
        if (not isinstance(snake, list) or not 1 <= len(snake) <= SNAKE_SIZE**2 or
                any(not _point(point, SNAKE_SIZE) for point in snake) or
                len({tuple(point) for point in snake}) != len(snake) or
                not isinstance(raw["heading"], str) or raw["heading"] not in DIRECTIONS):
            raise ValidationError("Invalid snake")
        if raw["food"] is not None and (not _point(raw["food"], SNAKE_SIZE) or raw["food"] in snake):
            raise ValidationError("Invalid food position")
    else:
        _integer(raw["lane"], "lane", 0, 2)
        hazards = raw["hazards"]
        if (not isinstance(hazards, list) or len(hazards) != 3 or
                any(not isinstance(row, list) or len(row) != 1 or type(row[0]) is not int or
                    row[0] not in (0, 1, 2) for row in hazards)):
            raise ValidationError("Invalid racing obstacles")
    return deepcopy(raw)


def _point(value, size):
    return (isinstance(value, list) and len(value) == 2 and
            all(type(item) is int and 0 <= item < size for item in value))


def _can_place(board, piece, x, y):
    return all(0 <= x + dx < BOARD_WIDTH and 0 <= y + dy < BOARD_HEIGHT and
               board[y + dy][x + dx] == "." for dx, dy in PIECES[piece])


def _legal_columns(board, piece):
    width = max(x for x, _ in PIECES[piece]) + 1
    return [x for x in range(BOARD_WIDTH - width + 1) if _can_place(board, piece, x, 0)]


def _choices(state):
    kind = state["kind"]
    if kind == "dino":
        return {"Jump": "Jump now.", "Keep running": "Run now."}
    if kind == "blocks":
        return {f"Column {x + 1}": f"Drop the {state['piece']} piece at column {x + 1} from the left"
                for x in _legal_columns(state["board"], state["piece"])}
    if kind == "snake":
        return {name: f"Move one tile {name.lower()}" for name in DIRECTIONS
                if name != OPPOSITES[state["heading"]]}
    lane = state["lane"]
    return {name: text for name, text in (
        ("Left", "Move one lane left"), ("Stay", "Stay in the current lane"),
        ("Right", "Move one lane right")) if 0 <= lane + {"Left": -1, "Stay": 0, "Right": 1}[name] <= 2}


def snapshot(state, event="", animation=None):
    state = _validated_state(state)
    kind = state["kind"]
    model_criteria = _choices(state) if not state["done"] else {}
    if kind == "dino":
        model_state = ("The jump is ready." if state["jump_cooldown_turns"] == 0 else
                       "The dinosaur is recovering from its previous jump and cannot jump this turn.")
        model_state += " " + ("The cactus reaches the dinosaur this turn." if state["distance"] <= state["speed"] else
                              "The cactus is still too far away to reach the dinosaur this turn.")
        instructions = ("Choose Jump only for an arriving cactus when the jump is ready. "
                        "Otherwise choose Keep running.")
        score_label = "Cacti cleared"
    elif kind == "blocks":
        model_state = {"board_top_to_bottom": state["board"], "piece": state["piece"],
                       "piece_cells": PIECES[state["piece"]]}
        instructions = ("Choose where to drop the current piece. Columns count from the left. "
                        "Filled rows clear; keep the stack low.")
        score_label = "Lines cleared"
    elif kind == "snake":
        head = state["snake"][0]
        food = state["food"]
        relative_food = None if food is None else {
            "horizontal": "left" if food[0] < head[0] else "right" if food[0] > head[0] else "same column",
            "vertical": "above" if food[1] < head[1] else "below" if food[1] > head[1] else "same row",
        }
        adjacent = {}
        for direction, (dx, dy) in DIRECTIONS.items():
            cell = [head[0] + dx, head[1] + dy]
            if not _point(cell, SNAKE_SIZE):
                adjacent[direction] = "wall"
            elif cell == food:
                adjacent[direction] = "food"
            elif cell in state["snake"][:-1]:
                adjacent[direction] = "body"
            elif cell == state["snake"][-1]:
                adjacent[direction] = "vacating_tail"
            else:
                adjacent[direction] = "empty"
        model_state = {"heading": state["heading"], "food_relative_to_head": relative_food}
        if food is not None:
            old_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            for direction in model_criteria:
                dx, dy = DIRECTIONS[direction]
                distance = abs(head[0] + dx - food[0]) + abs(head[1] + dy - food[1])
                toward = "closer to food" if distance < old_distance else "farther from food"
                model_criteria[direction] = (f"Move {direction.lower()}: destination is "
                                             f"{adjacent[direction]}; {toward}.")
        instructions = ("Choose a move that avoids walls and body cells, then gets closer to the food. "
                        "Prefer eating food when possible. A vacating tail cell is safe.")
        score_label = "Food collected"
    else:
        model_state = {"player_lane": state["lane"] + 1,
                       "blocked_lanes_next_three_rows": [[lane + 1 for lane in row]
                                                        for row in state["hazards"]]}
        instructions = ("Choose one lane move for the next row. Lanes are numbered 1 to 3 "
                        "from left to right; avoid the blocked lane in the first row.")
        score_label = "Rows survived"
    return {
        "state": state,
        "model_state": model_state,
        "question": None if state["done"] else {
            "type": "choice", "instructions": instructions, "criteria": model_criteria},
        "stats": {"turn": state["turn"], "limit": MAX_TURNS,
                  "score": state["score"], "score_label": score_label},
        "event": event,
        "animation": animation,
    }


def advance_game(raw, action):
    state = _validated_state(raw)
    if state["done"]:
        raise ValidationError("This game is finished; start a new run")
    if not isinstance(action, str) or action not in _choices(state):
        raise ValidationError("Action is not available in the current state")
    kind = state["kind"]
    state["turn"] += 1
    animation = None
    if kind == "dino":
        jumped = action == "Jump" and state["jump_cooldown_turns"] == 0
        blocked_jump = action == "Jump" and not jumped
        state["jump_cooldown_turns"] = 1 if jumped else 0
        animation = "jump" if jumped else None
        reaches_player = state["distance"] <= state["speed"]
        if reaches_player and not jumped:
            state["done"] = True
            event = ("Still landing; the jump could not start. " if blocked_jump else "")
            event += "The cactus reached the dinosaur. Game over."
        elif reaches_player:
            state["score"] += 1
            state["seed"], distance = _next(state["seed"], 13)
            state["distance"] = distance + state["speed"] + 1
            event = "Clean jump. Cactus cleared."
        else:
            state["distance"] -= state["speed"]
            event = ("Jumped too early; landing cooldown applies next turn." if jumped else
                     "Still landing; the jump could not start." if blocked_jump else "Safe. Keep going.")
    elif kind == "blocks":
        x = int(action.split()[1]) - 1
        y = 0
        while _can_place(state["board"], state["piece"], x, y + 1):
            y += 1
        board = [list(row) for row in state["board"]]
        for dx, dy in PIECES[state["piece"]]:
            board[y + dy][x + dx] = "#"
        rows = ["".join(row) for row in board]
        remaining = [row for row in rows if row != "#" * BOARD_WIDTH]
        cleared = BOARD_HEIGHT - len(remaining)
        state["board"] = ["." * BOARD_WIDTH] * cleared + remaining
        state["score"] += cleared
        state["seed"], index = _next(state["seed"], len(PIECES))
        state["piece"] = tuple(PIECES)[index]
        state["done"] = not _legal_columns(state["board"], state["piece"])
        event = f"Cleared {cleared} {'line' if cleared == 1 else 'lines'}." if cleared else "Piece placed."
        if state["done"]:
            event += " The stack reached the top. Game over."
    elif kind == "snake":
        dx, dy = DIRECTIONS[action]
        head = [state["snake"][0][0] + dx, state["snake"][0][1] + dy]
        eats = head == state["food"]
        body = state["snake"] if eats else state["snake"][:-1]
        if not _point(head, SNAKE_SIZE) or head in body:
            state["done"] = True
            event = "The snake hit a wall or its body. Game over."
        else:
            state["snake"] = [head] + body
            state["heading"] = action
            if eats:
                state["score"] += 1
                state["seed"], state["food"] = _food(state["seed"], state["snake"])
                state["done"] = state["food"] is None
                event = "Food collected."
            else:
                event = "Safe move."
    else:
        lane = state["lane"] + {"Left": -1, "Stay": 0, "Right": 1}[action]
        state["lane"] = lane
        if lane in state["hazards"][0]:
            state["done"] = True
            event = "The car hit an obstacle. Game over."
        else:
            state["score"] += 1
            state["seed"], blocked = _next(state["seed"], 3)
            state["hazards"] = state["hazards"][1:] + [[blocked]]
            event = "Clear lane. One row survived."
    if state["turn"] >= MAX_TURNS and not state["done"]:
        state["done"] = True
        event += " Run complete."
    return snapshot(state, event, animation)
