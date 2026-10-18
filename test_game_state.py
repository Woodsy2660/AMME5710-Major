"""
Tests for game_state.py and board_watcher.py. No images or video needed.

Run with:  python test_game_state.py   (inside the project venv)
"""

import config
from game_state import GameState, empty_potted
from board_watcher import BoardWatcher


def shot(red=0, yellow=0, black=False, cue=False):
    potted = empty_potted()
    potted['red'] = red
    potted['yellow'] = yellow
    potted['black'] = black
    potted['cue'] = cue
    return potted


# ---------------------------------------------------------------- rules

def test_first_pot_claims_colour():
    game = GameState()
    game.apply_shot(shot(red=1))
    assert game.groups == ['red', 'yellow']
    assert game.current == 0                     # potted, so keeps the turn
    assert game.balls_left(0) == 6


def test_potting_both_colours_leaves_table_open():
    game = GameState()
    game.apply_shot(shot(red=1, yellow=1))
    assert game.groups == [None, None]
    assert game.current == 0


def test_miss_passes_turn():
    game = GameState()
    game.apply_shot(shot())
    assert game.current == 1


def test_potting_opponent_ball_is_foul_and_gives_two_visits():
    game = GameState()
    game.apply_shot(shot(red=1))                 # player 0 is red
    event = game.apply_shot(shot(yellow=1))      # pots a yellow
    assert 'potted opponent ball' in event['fouls']
    assert game.current == 1
    assert game.visits_left == 2
    game.apply_shot(shot())                      # player 1 misses: uses the free visit
    assert game.current == 1
    game.apply_shot(shot())                      # misses again: turn passes
    assert game.current == 0


def test_cue_ball_potted_is_foul():
    game = GameState()
    event = game.apply_shot(shot(cue=True))
    assert event['fouls'] == ['cue ball potted']
    assert game.fouls[0] == 1
    assert game.current == 1


def test_black_after_clearing_wins():
    game = GameState()
    game.apply_shot(shot(red=7))
    assert game.balls_left(0) == 0
    event = game.apply_shot(shot(black=True))
    assert event['winner'] == 0
    assert game.winner == 0


def test_black_early_loses():
    game = GameState()
    game.apply_shot(shot(red=3))
    game.apply_shot(shot(black=True))
    assert game.winner == 1


def test_last_ball_and_black_together_loses():
    game = GameState()
    game.apply_shot(shot(red=6))
    game.apply_shot(shot(red=1, black=True))
    assert game.winner == 1


def test_no_shots_after_game_over():
    game = GameState()
    game.apply_shot(shot(black=True))
    assert game.apply_shot(shot(red=1)) is None


def test_overcount_is_capped():
    game = GameState()
    game.apply_shot(shot(red=9))
    assert game.remaining['red'] == 0


# ---------------------------------------------------------------- board watcher

def ball(colour, x, y):
    return {'colour': colour, 'map_x': x, 'map_y': y}


def rack():
    """A table with every ball on it, spread out along a grid."""
    balls = [ball('cue', 100, 200), ball('black', 600, 200)]
    for i in range(config.BALLS_PER_COLOUR):
        balls.append(ball('red', 400 + 30 * i, 100))
        balls.append(ball('yellow', 400 + 30 * i, 300))
    return balls


def moved_cue(balls, x=300):
    """The same table with the cue ball somewhere else (it was hit)."""
    new = [dict(b) for b in balls]
    new[0]['map_x'] = x
    return new


def test_first_state_is_not_a_shot():
    watcher = BoardWatcher()
    assert watcher.commit(rack(), shot_happened=True) is None


def test_pot_is_reported_with_pocket():
    watcher = BoardWatcher()
    watcher.commit(rack())
    after = [b for b in moved_cue(rack()) if not (b['colour'] == 'red' and b['map_x'] == 400)]
    result = watcher.commit(after, shot_happened=True)
    assert result['potted']['red'] == 1
    assert result['potted']['yellow'] == 0
    assert not result['potted']['cue']
    assert result['note'] == 'red TM'                 # (400, 100) is nearest the top-middle pocket


def test_miss_is_a_shot_with_nothing_potted():
    watcher = BoardWatcher()
    watcher.commit(rack())
    result = watcher.commit(moved_cue(rack()), shot_happened=True)
    assert result is not None
    assert result['potted'] == {'red': 0, 'yellow': 0, 'black': False, 'cue': False}


def test_no_motion_between_states_is_not_a_shot():
    # Someone leaned over the table (view blocked), then stepped back.
    watcher = BoardWatcher()
    watcher.commit(rack())
    assert watcher.commit(moved_cue(rack()), shot_happened=False) is None


def test_leaning_in_without_moving_balls_is_not_a_shot():
    watcher = BoardWatcher()
    watcher.commit(rack())
    assert watcher.commit(rack(), shot_happened=True) is None


def test_double_detected_ball_is_capped():
    watcher = BoardWatcher()
    doubled = rack() + [ball('red', 700, 150)]        # 8 reds seen
    watcher.commit(doubled)
    result = watcher.commit(moved_cue(rack()), shot_happened=True)
    assert result['potted']['red'] == 0


def test_cue_ball_scratch_then_placed_back():
    watcher = BoardWatcher()
    watcher.commit(rack())
    scratched = [dict(b) for b in rack()[1:]]
    scratched[0]['map_x'] = 650                        # black nudged
    result = watcher.commit(scratched, shot_happened=True)
    assert result['potted']['cue']

    placed = [ball('cue', 120, 220)] + scratched       # ball in hand: not a shot
    assert watcher.commit(placed, shot_happened=True) is None

    result = watcher.commit(moved_cue(placed), shot_happened=True)
    assert result is not None and not result['potted']['cue']


if __name__ == '__main__':
    tests = [name for name in dir() if name.startswith('test_')]
    failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"pass  {name}")
        except AssertionError:
            failed = failed + 1
            print(f"FAIL  {name}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
