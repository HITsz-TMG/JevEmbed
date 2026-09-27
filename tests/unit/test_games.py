"""The game demo should have reproducible rules and bounded states."""

import pytest

from jevembed import ValidationError
from jevembed.games import MAX_TURNS, advance_game, new_game, snapshot


@pytest.mark.parametrize("kind", ["dino", "blocks", "snake", "racing"])
def test_games_start_with_a_valid_choice_and_are_reproducible(kind):
    first = new_game(kind, 19)
    assert first == new_game(kind, 19)
    assert first["question"]["type"] == "choice"
    assert first["question"]["criteria"]
    action = next(iter(first["question"]["criteria"]))
    next_turn = advance_game(first["state"], action)
    assert next_turn["stats"]["turn"] == 1
    assert first["stats"]["turn"] == 0


def test_dino_jumps_only_when_the_cactus_reaches_the_player():
    state = new_game("dino")["state"]
    state["distance"] = state["speed"]
    cleared = advance_game(state, "Jump")
    assert cleared["state"]["score"] == 1
    assert not cleared["state"]["done"]
    collided = advance_game(state, "Keep running")
    assert collided["state"]["done"]
    assert collided["state"]["score"] == 0
    assert collided["question"] is None


def test_tetris_clears_two_rows_when_the_square_fills_the_gap():
    state = new_game("blocks")["state"]
    assert "Column 3" in snapshot(state)["question"]["criteria"]
    result = advance_game(state, "Column 3")
    assert result["state"]["score"] == 2
    assert result["state"]["board"] == ["......"] * 10
    assert "Cleared 2 lines" in result["event"]


def test_dino_early_jump_prevents_jumping_over_the_next_turns_cactus():
    state = new_game("dino")["state"]
    state["distance"] = state["speed"] + 1
    early = advance_game(state, "Jump")
    assert early["animation"] == "jump"
    assert early["state"]["jump_cooldown_turns"] == 1
    assert "cannot jump this turn" in early["model_state"]
    collision = advance_game(early["state"], "Jump")
    assert collision["state"]["done"]
    assert collision["animation"] is None
    assert "Game over" in collision["event"]


def test_dino_timed_jumps_can_survive_every_generated_obstacle():
    for seed in range(100):
        game = new_game("dino", seed)
        while not game["state"]["done"]:
            state = game["state"]
            action = "Jump" if state["distance"] <= state["speed"] else "Keep running"
            if action == "Jump":
                assert state["jump_cooldown_turns"] == 0
            game = advance_game(state, action)
        assert "Game over" not in game["event"]
        assert game["stats"]["turn"] == MAX_TURNS


def test_snake_can_eat_and_cannot_reverse_or_cross_a_wall():
    state = new_game("snake")["state"]
    state["food"] = [4, 4]
    assert "Left" not in snapshot(state)["question"]["criteria"]
    ate = advance_game(state, "Right")
    assert ate["state"]["score"] == 1
    assert len(ate["state"]["snake"]) == len(state["snake"]) + 1
    wall = {**state, "snake": [[7, 0], [6, 0]], "food": [0, 0]}
    assert advance_game(wall, "Right")["state"]["done"]


def test_racing_uses_the_immediate_obstacle_and_stays_within_three_lanes():
    state = new_game("racing")["state"]
    state["hazards"] = [[1], [0], [2]]
    assert advance_game(state, "Stay")["state"]["done"]
    moved = advance_game(state, "Left")
    assert moved["state"]["score"] == 1
    assert moved["state"]["lane"] == 0
    assert "Left" not in moved["question"]["criteria"]


def test_snake_observations_match_food_walls_and_vacating_tail():
    state = new_game("snake")["state"]
    state.update(snake=[[7, 0], [6, 0], [5, 0]], food=[7, 1], heading="Right")
    game = snapshot(state)
    observed = game["model_state"]
    assert observed["food_relative_to_head"] == {"horizontal": "same column", "vertical": "below"}
    assert "destination is wall" in game["question"]["criteria"]["Up"]
    assert "destination is wall" in game["question"]["criteria"]["Right"]
    assert "destination is food; closer to food" in game["question"]["criteria"]["Down"]
    state.update(snake=[[2, 2], [2, 3], [1, 3], [1, 2]], food=[7, 7], heading="Up")
    choices = snapshot(state)["question"]["criteria"]
    assert "destination is vacating_tail" in choices["Left"]
    assert "Down" not in choices
    assert not advance_game(state, "Left")["state"]["done"]


def test_game_state_and_actions_are_bounded():
    with pytest.raises(ValidationError, match="Unknown game"):
        new_game("unknown")
    with pytest.raises(ValidationError, match="seed"):
        new_game("dino", True)
    state = new_game("blocks")["state"]
    state["board"] = ["#" * 1000] * 1000
    with pytest.raises(ValidationError, match="board"):
        advance_game(state, "Column 1")
    state = new_game("dino")["state"]
    with pytest.raises(ValidationError, match="Action"):
        advance_game(state, "Fly")
    state["turn"] = MAX_TURNS + 1
    with pytest.raises(ValidationError, match="turn"):
        snapshot(state)
