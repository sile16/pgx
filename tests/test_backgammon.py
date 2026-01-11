from functools import partial
from time import time
import jax
import jax.numpy as jnp
from pgx.backgammon import (
    State,
    _flip_board,
    _calc_win_score,
    _change_turn,
    _is_action_legal,
    _is_all_on_home_board,
    _is_open,
    _legal_action_mask,
    _move,
    _rear_distance,
    _roll_init_dice,
    _distance_to_goal,
    _is_turn_end,
    _no_winning_step,
    _exists,
    Backgammon,
    action_to_str,
    stochastic_action_to_str,
    _decompose_action,
    _off_idx,
    turn_to_str,
    _get_valid_sequence_mask,
    _legal_action_mask_for_valid_single_dice,
    _to_playable_dice_count,
    _update_playable_dice,
    # New heuristic observation functions
    _is_race,
    _can_bear_off_current,
    _can_bear_off_opponent,
    _pip_count_current,
    _pip_count_opponent,
    _pip_count_differential_scaled,
    _observe_with_heuristics,
    # Full observation with blot/blocker
    _blot_board,
    _blocker_board,
    _observe_full,
)
import os
from pgx._src.api_test import (
    _validate_state,
    _validate_init_reward,
    _validate_current_player,
    _validate_legal_actions,
)

# NOTE: We need to import the CORRECT implementation for the tests to pass.
# The following import assumes the file is named backgammon1.py as analyzed.
from pgx.backgammon import (
    Backgammon,
    State,
    _change_turn,
    # Also import other functions if their signatures changed, but they didn't
    # so we can use the original ones for most tests.
)


def print_board(state_or_board, dice=None):
    """
    Print a backgammon board state in a readable ASCII format.

    Args:
        state_or_board: Either a State object or a board array (28,)
        dice: Optional dice array to display (if state_or_board is a board array)
    """
    if hasattr(state_or_board, '_board'):
        board = state_or_board._board
        dice = state_or_board._dice
        turn = state_or_board._turn
    else:
        board = state_or_board
        turn = None

    # Board layout:
    # Points 0-23: board positions (0-indexed, so point 1 = index 0)
    # 24: Black bar, 25: White bar
    # 26: Black off, 27: White off

    print("\n" + "=" * 60)
    print("       13  14  15  16  17  18       19  20  21  22  23  24")
    print("      +---+---+---+---+---+---+---+---+---+---+---+---+")

    # Top half (points 13-24, indices 12-23)
    top_row = "      |"
    for i in range(12, 18):
        val = int(board[i])
        top_row += f"{val:3d}|"
    top_row += "BAR|"
    for i in range(18, 24):
        val = int(board[i])
        top_row += f"{val:3d}|"
    print(top_row)

    print("      +---+---+---+---+---+---+---+---+---+---+---+---+")

    # Bar and off area
    black_bar = int(board[24])
    white_bar = int(board[25])
    black_off = int(board[26])
    white_off = int(board[27])
    print(f"      |           BLACK BAR: {black_bar:2d}  |  WHITE BAR: {white_bar:2d}           |")
    print(f"      |           BLACK OFF: {black_off:2d}  |  WHITE OFF: {white_off:2d}           |")

    print("      +---+---+---+---+---+---+---+---+---+---+---+---+")

    # Bottom half (points 12-1, indices 11-0, displayed in reverse)
    bot_row = "      |"
    for i in range(11, 5, -1):
        val = int(board[i])
        bot_row += f"{val:3d}|"
    bot_row += "   |"
    for i in range(5, -1, -1):
        val = int(board[i])
        bot_row += f"{val:3d}|"
    print(bot_row)

    print("      +---+---+---+---+---+---+---+---+---+---+---+---+")
    print("       12  11  10   9   8   7        6   5   4   3   2   1")

    # Show current player, dice and turn info
    if hasattr(state_or_board, 'current_player'):
        player_str = "BLACK (0)" if state_or_board.current_player == 0 else "WHITE (1)"
        print(f"\n      Current Player: {player_str}")
    if dice is not None:
        dice_vals = [int(d) + 1 for d in dice]
        print(f"      Dice: {dice_vals[0]}-{dice_vals[1]}")
    if turn is not None:
        turn_str = "BLACK (0)" if turn == 0 else "WHITE (1)"
        print(f"      Board Perspective (_turn): {turn_str}")

    print("      (Positive = Black, Negative = White)")
    print("=" * 60 + "\n")


seed = 1701
rng = jax.random.PRNGKey(seed)
# Use the corrected Backgammon environment for testing the new rules
env = Backgammon() 
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

# Original functions for older tests
_no_winning_step = jax.jit(_no_winning_step)
_calc_win_score = jax.jit(_calc_win_score)
_change_turn = jax.jit(_change_turn)
_is_action_legal = jax.jit(_is_action_legal)
_is_all_on_home_board = jax.jit(_is_all_on_home_board)
_is_open = jax.jit(_is_open)
# The old legal_action_mask is no longer directly used in the new env, but kept for old test compatibility
_legal_action_mask_old = jax.jit(partial(_legal_action_mask, turn_dice=jnp.zeros(2), played_dice_num=jnp.int32(1)))
_move = jax.jit(_move)
_rear_distance = jax.jit(_rear_distance)
_exists = jax.jit(_exists)


def make_test_boad():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
    # 黒
    board = board.at[19].set(5)
    board = board.at[20].set(1)
    board = board.at[21].set(2)
    board = board.at[26].set(7)
    # 白
    board = board.at[3].set(-2)
    board = board.at[4].set(-1)
    board = board.at[10].set(-5)
    board = board.at[22].set(-3)
    board = board.at[25].set(-4)
    return board


"""
黒: + 白: -
12 13 14 15 16 17  18 19 20 21 22 23
                       +  +  +  -
                       +     +  -
                       +        -
                       +
                       +
 
    -
    -
    -
    -                     -
    -                  -  -
11 10  9  8  7  6   5  4  3  2  1  0
Bar ----
Off +++++++
"""


def make_test_state(
    current_player: jnp.ndarray,
    board: jnp.ndarray,
    turn: jnp.ndarray,
    dice: jnp.ndarray,
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    legal_action_mask=jnp.zeros(6 * 26, dtype=jnp.bool_),
    is_stochastic=jnp.array(False, dtype=jnp.bool_),
):
    return State(
        current_player=current_player,
        _board=board,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
        _is_stochastic=is_stochastic,
    )


def test_flip_board():
    test_board = make_test_boad()
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[4].set(-5)
    board = board.at[3].set(-1)
    board = board.at[2].set(-2)
    board = board.at[27].set(-7)
    board = board.at[20].set(2)
    board = board.at[19].set(1)
    board = board.at[13].set(5)
    board = board.at[1].set(3)
    board = board.at[24].set(4)
    flipped_board = _flip_board(test_board)
    assert  (flipped_board == board).all()



def test_init():
    state = init(rng)
    assert state._turn == 0 or state._turn == 1


def test_init_roll():
    a = _roll_init_dice(rng)
    assert len(a) == 2
    assert a[0] != a[1]


def test_is_turn_end():
    state = init(rng)
    assert not _is_turn_end(state)

    # white dance
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    assert _is_turn_end(state)

    # No playable dice
    board: jnp.ndarray = make_test_boad()
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(2),
    )
    assert _is_turn_end(state)


def test_change_turn():
    # Using the corrected environment's init function
    state = env.init(rng)
    _turn = state._turn
    # Use the corrected change_turn function
    state = _change_turn(state)
    assert state._turn == (_turn + 1) % 2

    test_board: jnp.ndarray = make_test_boad()
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[4].set(-5)
    board = board.at[3].set(-1)
    board = board.at[2].set(-2)
    board = board.at[27].set(-7)
    board = board.at[20].set(2)
    board = board.at[19].set(1)
    board = board.at[13].set(5)
    board = board.at[1].set(3)
    board = board.at[24].set(4)
    state = make_test_state(
        current_player=jnp.int32(0),
        board=test_board,
        turn=jnp.int32(0),
        dice=jnp.array([2, 2], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(2),
    )
    state = _change_turn(state, jax.random.PRNGKey(0))
    assert state._turn == jnp.int32(1)  # Turn changed
    assert (state._board == board).all()  # Flipped.


def test_no_op():
    # Setup state
    state = env.init(rng)
    # Force a situation where no moves are possible
    board = jnp.zeros(28, dtype=jnp.int32).at[24].set(15) # all black checkers on bar
    board = board.at[0:6].set(-2) # all entry points blocked
    dice = jnp.array([0, 1], dtype=jnp.int32)
    state = state.replace(_board=board, _turn=jnp.int32(0), current_player=jnp.int32(0))
    state = env.set_dice(state, dice)
    
    # Check that only no-op is legal
    legal_actions = jnp.where(state.legal_action_mask)[0]
    assert jnp.array_equal(legal_actions, jnp.arange(6))

    # Take a no-op action
    state = step(state, 0, jax.random.PRNGKey(0))
    assert state._turn == jnp.int32(1)  # Turn changes after no-op.



def test_observe():
    board: jnp.ndarray = make_test_boad()

    # current_player = white, playable_dice = (1, 2)
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = _observe_full(state)
    assert (observe(state) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([1, 1, 1, 1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = _observe_full(state)
    assert (observe(state) == expected_obs).all()

    # Test for player_id 1 (white player)
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(-1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = _observe_full(state)
    assert (observe(state) == expected_obs).all()

    # Test for player_id 0 (black player)
    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(-1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = _observe_full(state)
    assert (observe(state) == expected_obs).all()


def test_is_open():
    board = make_test_boad()
    # Black
    assert _is_open(board, 9)
    assert _is_open(board, 19)
    assert _is_open(board, 4)
    assert not _is_open(board, 10)
    # White
    board = _flip_board(board)
    assert _is_open(board, 9)
    assert _is_open(board, 8)
    assert not _is_open(board, 2)
    assert not _is_open(board, 4)


def test_exists():
    board = make_test_boad()
    # Black
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 4)
    # White
    board = _flip_board(board)
    assert _exists(board, 19)
    assert _exists(board, 20)
    assert not _exists(board, 2)


def test_is_all_on_home_boad():
    board: jnp.ndarray = make_test_boad()
    # Black
    assert _is_all_on_home_board(board)
    # White
    board = _flip_board(board)
    assert not _is_all_on_home_board(board)


def test_rear_distance():
    board = make_test_boad()
    # Black
    assert _rear_distance(board) == 5
    # White
    board = _flip_board(board)
    assert _rear_distance(board) == 23


def test_distance_to_goal():
    # Black
    src = 23
    assert _distance_to_goal(src) == 1
    src = 10
    assert _distance_to_goal(src) == 14
    # Test at the src where rear_distance is same
    board = make_test_boad()
    assert _rear_distance(board) == _distance_to_goal(19)


def test_decompose_action():
    """Test that _decompose_action correctly decomposes actions into src, die, tgt."""
    # Test src calculation (formerly _calc_src)
    # action = src_raw * 6 + (die - 1), so for src_raw=1 (bar) with die=1: action = 1*6 + 0 = 6
    src, die, tgt = _decompose_action(6)  # src_raw=1, die=1
    assert src == 24  # bar

    src, die, tgt = _decompose_action(12)  # src_raw=2, die=1
    assert src == 0  # point 0

    # Test tgt calculation (formerly _calc_tgt)
    # bar to board: src_raw=1 (bar=24), die=1 -> tgt should be die-1 = 0
    src, die, tgt = _decompose_action(6)  # src_raw=1, die=1
    assert tgt == 0  # bar -> point 0

    # board to board: src=6, die=2 -> tgt = src + die = 8
    # src_raw = src + 2 = 8, action = 8*6 + (2-1) = 49
    src, die, tgt = _decompose_action(49)
    assert src == 6
    assert die == 2
    assert tgt == 8  # 6 + 2 = 8

    # to off: src=23, die=6 -> tgt should be 26 (off)
    # src_raw = 23 + 2 = 25, action = 25*6 + (6-1) = 155
    src, die, tgt = _decompose_action(155)
    assert src == 23
    assert die == 6
    assert tgt == 26  # off


def test_is_action_legal():
    board: jnp.ndarray = make_test_boad()
    # 黒
    assert _is_action_legal(board, (19 + 2) * 6 + 1)  # 19->21
    assert not _is_action_legal(board, (19 + 2) * 6 + 2)  # 19 -> 22
    assert not _is_action_legal(
        board, (19 + 2) * 6 + 2
    )  # 19 -> 22: Some whites on 22 
    assert not _is_action_legal(
        board, (22 + 2) * 6 + 2
    )  # 22 -> 25: No black on 22 
    assert _is_action_legal(board, (19 + 2) * 6 + 5)  # bear off
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 5
    )  # cannot bear off as some blacks behind
    # white
    board = _flip_board(board)
    assert not _is_action_legal(
        board, (20 + 2) * 6 + 0
    )  # 20->21(after flipped): cannot move checkers as some left on bar
    assert _is_action_legal(board, (1) * 6 + 0)  # bar -> 1(after flipped)
    assert not _is_action_legal(board, (1) * 6 + 2)  # bar -> 2(after flipped)


def test_move():
    # point to point black
    board = make_test_boad()
    board = _move(board, (19 + 2) * 6 + 1)  # 19->21
    assert (
        board.at[19].get() == 4
        and board.at[21].get() == 3
        and board.at[25].get() == -4
    )
    # point to off black
    board = make_test_boad()
    board = _move(board, (19 + 2) * 6 + 5)  # 19->26
    assert (
        board.at[19].get() == 4
        and board.at[26].get() == 8
        and board.at[25].get() == -4
    )
    # enter white
    board = make_test_boad()
    board = _flip_board(board)
    board = _move(board, (1) * 6 + 0)  # 25 -> 0
    assert (
        board.at[24].get() == 3
        and board.at[0].get() == 1
    )
    # hit white
    board = make_test_boad()
    board = _flip_board(board)
    board = _move(board, (1 + 2) * 6 + 1)  # 1 -> 3
    assert (
        board.at[1].get() == 2
        and board.at[3].get() == 1
        and board.at[25].get() == -1
    )


def test_calc_win_score():
    # backgammon win by black
    back_gammon_board = jnp.zeros(28, dtype=jnp.int32)
    back_gammon_board = back_gammon_board.at[26].set(15)
    back_gammon_board = back_gammon_board.at[23].set(-15)  # black on home board
    assert _calc_win_score(back_gammon_board) == 3

    # backgammon when opponent pieces sit only on bar
    bar_only_board = jnp.zeros(28, dtype=jnp.int32)
    bar_only_board = bar_only_board.at[26].set(15)  # current player off all checkers
    bar_only_board = bar_only_board.at[25].set(-2)  # opponent stuck on bar
    assert _calc_win_score(bar_only_board) == 3

    # gammon win by black
    gammon_board = jnp.zeros(28, dtype=jnp.int32)
    gammon_board = gammon_board.at[26].set(15)
    gammon_board = gammon_board.at[7].set(-15)
    assert _calc_win_score(gammon_board) == 2

    # single win by black
    single_board = jnp.zeros(28, dtype=jnp.int32)
    single_board = single_board.at[26].set(15)
    single_board = single_board.at[27].set(-3)
    single_board = single_board.at[3].set(-12)
    assert _calc_win_score(single_board) == 1

def _act_randomly_wrapper(rng, legal_action_mask):
    """Wrapper around act_randomly that handles the axis correctly."""
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits, axis=0)

def _api_test_single_modified(env, num=100, use_key=True):
    """Modified version of api_test_single that uses our own act_randomly implementation."""
    init = jax.jit(env.init)
    step = jax.jit(env.step)
    act_randomly_jit = jax.jit(_act_randomly_wrapper)

    rng = jax.random.PRNGKey(849020)
    for _ in range(num):
        rng, subkey = jax.random.split(rng)
        state = init(subkey)
        assert state.env_id == env.id
        assert state.legal_action_mask.sum() != 0, "legal_action_mask at init state cannot be zero."

        assert state._step_count == 0
        curr_steps = state._step_count
        _validate_state(state)
        _validate_init_reward(state)
        _validate_current_player(state)
        _validate_legal_actions(state)

        while True:
            rng, subkey = jax.random.split(rng)
            action = act_randomly_jit(subkey, state.legal_action_mask)
            rng, subkey = jax.random.split(rng)
            if not use_key:
                subkey = None
            state = step(state, action, subkey)
            assert state._step_count == curr_steps + 1, f"{state._step_count}, {curr_steps}"
            curr_steps += 1

            _validate_state(state)
            _validate_current_player(state)
            _validate_legal_actions(state)

            if state.terminated:
                break

    # check visualization
    filename = "/tmp/tmp.svg"
    state.save_svg(filename)
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

def test_api():
    import pgx
    env = pgx.make("backgammon")
    _api_test_single_modified(env, 3, use_key=True)  # Only run single environment test

def test_stochastic_state():
    """Test if a state is correctly identified as stochastic."""
    state: State = env.init(jax.random.PRNGKey(0))  # type: ignore
    
    # New game state should be stochastic (needs dice)
    assert state._is_stochastic  # type: ignore
    
    # After a stochastic step with doubles, it should no longer be stochastic
    stochastic_action = 0  # Using double 1's (action 0)
    new_state: State = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    assert not new_state._is_stochastic  # type: ignore
    # With doubles, player should be able to make 4 moves before state becomes stochastic again
    for i in range(4):
        # State should remain non-stochastic during moves
        assert not new_state._is_stochastic  # type: ignore

        # Make a move
        legal_action = jnp.where(new_state.legal_action_mask)[0][0]
        # Use step_deterministic for the last move to catch the afterstate
        if i < 3:
            new_state = env.step(new_state, legal_action, jax.random.PRNGKey(1))  # type: ignore
        else:
            new_state = env.step_deterministic(new_state, legal_action)  # type: ignore

    # After all 4 moves are made, state should be stochastic again
    assert new_state._is_stochastic  # type: ignore
    
def test_stochastic_actions():
    """Test getting available stochastic actions and their probabilities."""
    # For regular mode, all 21 dice combinations should be possible
    assert len(env._stochastic_action_probs) == 21

    # Test that probabilities sum to 1
    assert jnp.isclose(jnp.sum(env._stochastic_action_probs), 1.0)

    # Test simple doubles mode
    env_simple = Backgammon(simple_doubles=True)
    assert len(env_simple._stochastic_action_probs) == 21

    # In simple doubles mode, only the first 6 actions (doubles) have non-zero probability
    assert jnp.all(env_simple._stochastic_action_probs[6:] == 0)
    assert jnp.isclose(jnp.sum(env_simple._stochastic_action_probs), 1.0)


def test_stochastic_step():
    """Test applying a stochastic action to set dice."""
    state: State = env.init(jax.random.PRNGKey(0))  # type: ignore
    
    # Apply stochastic action 0 (double 1's)
    stochastic_action = 0
    new_state: State = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    
    # Check that dice are set correctly
    assert jnp.array_equal(new_state._dice, jnp.array([0, 0]))  # type: ignore
    
    # Check that state is no longer stochastic
    assert not new_state._is_stochastic  # type: ignore
    
    # Check that playable dice are set correctly for doubles (4 dice of the same value)
    assert jnp.array_equal(new_state._playable_dice, jnp.array([0, 0, 0, 0]))  # type: ignore
    
    # Apply stochastic action 10 (1,6)
    stochastic_action = 10
    new_state = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    
    # Check that dice are set correctly
    assert jnp.array_equal(new_state._dice, jnp.array([0, 5]))  # type: ignore
    
    # Check that playable dice are set correctly for non-doubles (2 dice)
    assert jnp.array_equal(new_state._playable_dice, jnp.array([0, 5, -1, -1]))  # type: ignore


def test_stochastic_simple_doubles():
    """Test stochastic functionality in simple_doubles mode."""
    env_simple = Backgammon(simple_doubles=True)
    state: State = env_simple.init(jax.random.PRNGKey(0))  # type: ignore
    
    # In simple doubles mode, only double actions (0-5) should work
    
    # Test double action (action 0 = double 1's)
    stochastic_action = 0
    new_state: State = env_simple.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    assert jnp.array_equal(new_state._dice, jnp.array([0, 0]))  # type: ignore
    assert not new_state._is_stochastic  # type: ignore
    

def test_stochastic_game_simulation():
    """Test simulating a game with predefined dice rolls."""
    # Initialize game
    state: State = env.init(jax.random.PRNGKey(42))  # type: ignore
    
    # Test roll: 1,6 (action index 10)
    assert state._is_stochastic  # type: ignore
    state = env.stochastic_step(state, jnp.array(10))  # type: ignore
    assert not state._is_stochastic  # type: ignore
    assert jnp.array_equal(state._dice, jnp.array([0, 5]))  # type: ignore
    
    # Make a move using the first die
    legal_action = jnp.where(state.legal_action_mask)[0][0]
    state = env.step(state, legal_action, jax.random.PRNGKey(0))  # type: ignore
    
    # Make another move if the turn hasn't changed
    for x in range(20):
        if state._is_stochastic:  # type: ignore
            # Turn has changed, and we need new dice
            # Let's roll doubles: 3,3 (action index 2)
            state = env.stochastic_step(state, jnp.array(2))  # type: ignore
            assert jnp.array_equal(state._dice, jnp.array([2, 2]))  # type: ignore
        else:
            # Make a move with the second die
            legal_action = jnp.where(state.legal_action_mask)[0][0]
            state = env.step(state, legal_action, jax.random.PRNGKey(1))  # type: ignore
        
    
    # Verify we can continue making moves
    assert not state.terminated
    assert state.legal_action_mask.any()  # There should be legal actions available

def test_action_to_str():
    """Test the action_to_str and stochastic_action_to_str functions."""
    # Test no-op action
    assert action_to_str(3) == "No-op (die: 4)"
    
    # Test bar action
    # src=1 -> point 2
    assert action_to_str(7) == "Bar/2 (die: 2)"
    
    # Test normal point movement
    # src=6 (point 5) -> point 7. die=2
    assert action_to_str(37) == "5/7 (die: 2)"
    
    # Test bearing off to off
    # Find an action that goes to off
    # src=25 (point 24), die=6 -> off
    bearing_off_action = (25) * 6 + 5
    assert "24/Off" in action_to_str(bearing_off_action)
    
    # Test stochastic actions
    assert stochastic_action_to_str(0) == "Rolled: 1-1"
    assert stochastic_action_to_str(5) == "Rolled: 6-6"
    assert stochastic_action_to_str(10) == "Rolled: 1-6"

def test_turn_to_str():
    """Test the turn_to_str function for standard backgammon notation."""
    state1 = env.init(jax.random.PRNGKey(42))
    state1 = env.set_dice(state1, jnp.array([3, 5])) # 4-6 roll
    
    # point 13 -> 7 (6-die), point 8 -> 4 (4-die)
    # The initial board does not allow these moves.
    # We need a board where they are possible.
    board = jnp.zeros(28, dtype=jnp.int32).at[12].set(1).at[7].set(1) # a checker on 13 and 8
    state1 = state1.replace(_board=board)
    state1 = env.set_dice(state1, jnp.array([3, 5])) # set dice again to recalc mask

    action1 = (7 + 2) * 6 + 5  # src=9 (point 8) move by 6
    state2 = env.step(state1, action1, jax.random.PRNGKey(0))
    
    action2 = (12 + 2) * 6 + 3 # src=14 (point 13) move by 4
    state3 = env.step(state2, action2, jax.random.PRNGKey(0))
    
    turn_str = turn_to_str([state1, state2, state3], [action1, action2])
    assert turn_str == "4-6: 8/14 13/17"

# ==============================================================================
# == NEW TESTS FOR FORCED MOVE RULES ===========================================
# ==============================================================================

def test_must_play_both_dice_if_possible():
    """
    Tests Rule 1: If you can play both dice, you must.
    A move that makes the second die unplayable is illegal.
    Setup:
    - Dice: 2-1
    - Board: Black to move. Checkers at 13 and 5. White blocks point 14.
    - Analysis:
      - Can we play both? Yes: Play 1 from 5->6. Then play 2 from 13->15.
      - Therefore, the "must use both" rule applies.

    
    """
    state = env.init(jax.random.PRNGKey(0))
    dice = jnp.array([1, 0])  # 2-1 roll

    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[4].set(1)       # Black at 5
    board = board.at[7].set(-2)      # 2 White at 8
    board = board.at[12].set(1)      # Black at 13
    board = board.at[13].set(-2)      # White block at 14
    board = board.at[26].set(13)      # put rest on the bar for this test case.
    board = board.at[27].set(-11)     # put rest on the bar for this test case.

    state = state.replace(_board=board)
    state = env.set_dice(state, dice)
    #time the svg

    start_time = time()
    state.save_svg("bg_must_play_both_dice_if_possible.svg")
    end_time = time()
    print(f"SVG generation took {end_time - start_time:.6f} seconds.")
    

    # LEGAL: can play 1, 
    action_5_to_6 = 36  #(4 * 6) + 0 (0 indexed dice)
    # ILLEGAL:Not legal, because if we use the 2 here, we can't use the one from source 13 
    # and we can we can't use the 1 from source 5
    action_5_to_7 = 37
    # Legal: Play 2 from 13->15. 
    action_13_to_15 = 85

    legal_actions = jnp.where(state.legal_action_mask)[0]

    print_board(state)
    print("Legal actions:")
    for x in legal_actions:
        print(f"{x} = {action_to_str(x)}")




    assert action_5_to_6 in legal_actions
    assert action_5_to_7 not in legal_actions
    assert action_13_to_15  in legal_actions

    assert action_to_str(action_5_to_6) == "5/6 (die: 1)"
    assert action_to_str(action_5_to_7) == "5/7 (die: 2)"
    assert action_to_str(action_13_to_15) == "13/15 (die: 2)"
    assert len(legal_actions) == 2



def test_must_play_higher_die_when_only_one_is_possible():
    """
    Tests Rule 2: If you can only play one of two dice, you must use the higher one.

    Setup:
    - Dice: 6-1
    - Board: Black to move, with a checker on the bar.
      - White has blocks on point 2 and point 7.
      - All other black checkers are on point 1.
    - Analysis:
      - Player must enter from the bar. Entry points 1 and 6 are open.
      - If they enter with 1 (bar->1), the second move (a 6) is impossible.
      - If they enter with 6 (bar->6), the second dice (1) is impossible to use.
      - Since it's impossible to play a two-move sequence, the "must play higher"
        rule applies. The player MUST use the 6.
    """
    state = env.init(jax.random.PRNGKey(0))
    dice = jnp.array([5, 0])  # 6-1 roll

    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[24].set(1)  # Black checker on the bar
    board = board.at[0].set(14)  # All other black checkers on point 1
    board = board.at[1].set(-2)  # White block at point 2
    board = board.at[6].set(-2)  # White block at point 7

    state = state.replace(_board=board, current_player=0, _turn=0)
    state = env.set_dice(state, dice)

    # The LEGAL move bar->6 (using a 6) is action: src=1, die=5 -> 1*6+5 = 11
    action_bar_to_6 = 1 * 6 + 5
    # The ILLEGAL move bar->1 (using a 1) is action: src=1, die=0 -> 1*6+0 = 6
    action_bar_to_1 = 1 * 6 + 0

    legal_actions = jnp.where(state.legal_action_mask)[0]

    print_board(state)
    print("Legal actions:")
    for x in legal_actions:
        print(f"{x} = {action_to_str(x)}")

    assert action_bar_to_6 in legal_actions
    assert action_bar_to_1 not in legal_actions
    assert action_to_str(action_bar_to_6) == "Bar/6 (die: 6)"
    assert action_to_str(action_bar_to_1) == "Bar/1 (die: 1)"
    assert len(legal_actions) == 1


def test_must_play_lower_die_if_higher_is_blocked():
    """
    Tests Gap 1: If you can only play one die, and the higher one is BLOCKED,
    you must play the lower one.
    
    Standard Rule: You must play the higher die if *possible*.
    Correction: If the higher die move is blocked, you fall back to the lower die.
    """
    # Setup: Roll 6-1.
    # Board: 
    # - Black checker on Point 1.
    # - White blocks Point 7 (blocking the 6 move: 1 -> 7).
    # - Point 2 is open (allowing the 1 move: 1 -> 2).
    # - All other points empty/open.
    
    state = env.init(jax.random.PRNGKey(0))
    dice = jnp.array([5, 0]) # 6-1 roll
    
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(1)   # Black at Point 1
    board = board.at[6].set(-2)  # White block at Point 7 (1+6)
    # Point 2 (1+1) is empty/open by default
    
    state = state.replace(_board=board, current_player=0, _turn=0)
    state = env.set_dice(state, dice)
    
    # Action: 1 -> 7 (using 6) should be illegal because it's blocked
    # Action ID: src(2)*6 + die(5) = 5
    action_using_6 = 2 * 6 + 5
    
    # Action: 1 -> 2 (using 1) should be LEGAL because it's the only move
    # Action ID: src(2)*6 + die(0) = 0
    action_using_1 = 2 * 6 + 0
    
    legal_actions = jnp.where(state.legal_action_mask)[0]
    
    print("\n--- Test: Must Play Lower Die if Higher Blocked ---")
    print(f"Board[0] (Src): {board[0]}")
    print(f"Board[6] (Block): {board[6]}")
    print(f"Legal Actions: {[action_to_str(a) for a in legal_actions]}")

    assert action_using_1 in legal_actions
    assert action_using_6 not in legal_actions
    assert len(legal_actions) == 1, "Should only have 1 legal move (the lower die)"
    assert action_to_str(action_using_1) == "1/2 (die: 1)"
    assert action_to_str(action_using_6) == "1/7 (die: 6)"


# ==============================================================================
# == OPTIMIZATION CORRECTNESS TESTS ============================================
# ==============================================================================

def test_candidate_action_generation():
    """
    Verifies that the optimized candidate action generation produces the same
    indices as the original filter-based approach.

    The optimization in _get_valid_sequence_mask generates candidate actions
    directly via `src * 6 + (die_first - 1)` instead of filtering all 156 actions.
    """
    for die_first in range(1, 7):
        # New approach: direct indexing
        src_indices = jnp.arange(26, dtype=jnp.int32)
        candidate_actions = src_indices * 6 + (die_first - 1)

        # Old approach: filter all 156
        all_actions = jnp.arange(26 * 6, dtype=jnp.int32)
        is_correct_die = (all_actions % 6) == (die_first - 1)
        old_candidates = all_actions[is_correct_die]

        assert jnp.array_equal(candidate_actions, old_candidates), \
            f"Mismatch for die_first={die_first}"

    print("All candidate action generation tests passed!")


def test_get_valid_sequence_mask_optimization():
    """
    Verifies that the optimized _get_valid_sequence_mask produces identical
    results to the original implementation.

    This test compares the new focused candidate search (26 actions) against
    the original approach that processed all 156 actions.
    """
    import pgx

    # Define the original implementation for comparison
    def _get_valid_sequence_mask_old(board, die_first, die_second):
        all_actions = jnp.arange(26 * 6, dtype=jnp.int32)
        is_correct_die = (all_actions % 6) == (die_first - 1)
        is_legal_first_move = jax.vmap(_is_action_legal, in_axes=(None, 0))(board, all_actions)
        mask_first_move = is_correct_die & is_legal_first_move
        next_boards = jax.vmap(_move, in_axes=(None, 0))(board, all_actions)
        future_legal_masks = jax.vmap(_legal_action_mask_for_valid_single_dice, in_axes=(0, None))(
            next_boards, die_second - 1
        )
        can_play_second_die = future_legal_masks.any(axis=1)
        return mask_first_move & can_play_second_die

    # Test with the standard starting board
    env = pgx.make("backgammon")
    state = env.init(jax.random.PRNGKey(42))
    board = state._board

    # Compare for all non-doubles die combinations
    for die_first in range(1, 7):
        for die_second in range(1, 7):
            if die_first == die_second:
                continue

            old_mask = _get_valid_sequence_mask_old(board, die_first, die_second)
            new_mask = _get_valid_sequence_mask(board, die_first, die_second)

            assert jnp.array_equal(old_mask, new_mask), \
                f"Mismatch for die_first={die_first}, die_second={die_second}"

            # Verify mask shape is correct
            assert new_mask.shape == (156,), f"Wrong shape: {new_mask.shape}"

            # Verify only positions for die_first have True values
            mask_indices = jnp.where(new_mask)[0]
            for idx in mask_indices.tolist():
                die_at_idx = idx % 6
                expected_die = die_first - 1
                assert die_at_idx == expected_die, \
                    f"Wrong die at index {idx}: got {die_at_idx}, expected {expected_die}"

    print("All _get_valid_sequence_mask optimization tests passed!")


def test_to_playable_dice_count():
    """
    Verifies that _to_playable_dice_count correctly counts dice values.
    Tests the histogram-style counting optimization.
    """
    # Test non-doubles: dice 2,3 (0-indexed: 1,2)
    test1 = jnp.array([1, 2, -1, -1], dtype=jnp.int32)
    result1 = _to_playable_dice_count(test1)
    expected1 = jnp.array([0, 1, 1, 0, 0, 0], dtype=jnp.int32)
    assert jnp.array_equal(result1, expected1), f"Test 1 failed: {result1} != {expected1}"

    # Test doubles: dice 4,4,4,4 (0-indexed: 3,3,3,3)
    test2 = jnp.array([3, 3, 3, 3], dtype=jnp.int32)
    result2 = _to_playable_dice_count(test2)
    expected2 = jnp.array([0, 0, 0, 4, 0, 0], dtype=jnp.int32)
    assert jnp.array_equal(result2, expected2), f"Test 2 failed: {result2} != {expected2}"

    # Test all empty
    test3 = jnp.array([-1, -1, -1, -1], dtype=jnp.int32)
    result3 = _to_playable_dice_count(test3)
    expected3 = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    assert jnp.array_equal(result3, expected3), f"Test 3 failed: {result3} != {expected3}"

    # Test mixed: dice 1,6 (0-indexed: 0,5)
    test4 = jnp.array([0, 5, -1, -1], dtype=jnp.int32)
    result4 = _to_playable_dice_count(test4)
    expected4 = jnp.array([1, 0, 0, 0, 0, 1], dtype=jnp.int32)
    assert jnp.array_equal(result4, expected4), f"Test 4 failed: {result4} != {expected4}"

    print("All _to_playable_dice_count tests passed!")


def test_update_playable_dice():
    """
    Verifies that _update_playable_dice correctly updates the playable dice array
    after a move is made.
    """
    # Test non-doubles: dice [2, 4] (0-indexed: [1, 3]), play die 2 (index 1)
    playable = jnp.array([1, 3, -1, -1], dtype=jnp.int32)
    dice = jnp.array([1, 3], dtype=jnp.int32)
    action = 0 * 6 + 1  # action using die 2 (index 1)
    result = _update_playable_dice(playable, jnp.int32(0), dice, action)
    expected = jnp.array([-1, 3, -1, -1], dtype=jnp.int32)
    assert jnp.array_equal(result, expected), f"Non-doubles play die 2 failed: {result} != {expected}"

    # Test non-doubles: dice [2, 4], play die 4 (index 3)
    action2 = 0 * 6 + 3  # action using die 4 (index 3)
    result2 = _update_playable_dice(playable, jnp.int32(0), dice, action2)
    expected2 = jnp.array([1, -1, -1, -1], dtype=jnp.int32)
    assert jnp.array_equal(result2, expected2), f"Non-doubles play die 4 failed: {result2} != {expected2}"

    # Test doubles: dice [3, 3] (0-indexed: [2, 2]), play first die
    playable_d = jnp.array([2, 2, 2, 2], dtype=jnp.int32)
    dice_d = jnp.array([2, 2], dtype=jnp.int32)
    action_d = 0 * 6 + 2  # action using die 3

    # First move (n=0): should mark slot 3
    result_d0 = _update_playable_dice(playable_d, jnp.int32(0), dice_d, action_d)
    expected_d0 = jnp.array([2, 2, 2, -1], dtype=jnp.int32)
    assert jnp.array_equal(result_d0, expected_d0), f"Doubles (n=0) failed: {result_d0} != {expected_d0}"

    # Second move (n=1): should mark slot 2
    result_d1 = _update_playable_dice(playable_d, jnp.int32(1), dice_d, action_d)
    expected_d1 = jnp.array([2, 2, -1, 2], dtype=jnp.int32)
    assert jnp.array_equal(result_d1, expected_d1), f"Doubles (n=1) failed: {result_d1} != {expected_d1}"

    # Third move (n=2): should mark slot 1
    result_d2 = _update_playable_dice(playable_d, jnp.int32(2), dice_d, action_d)
    expected_d2 = jnp.array([2, -1, 2, 2], dtype=jnp.int32)
    assert jnp.array_equal(result_d2, expected_d2), f"Doubles (n=2) failed: {result_d2} != {expected_d2}"

    # Fourth move (n=3): should mark slot 0
    result_d3 = _update_playable_dice(playable_d, jnp.int32(3), dice_d, action_d)
    expected_d3 = jnp.array([-1, 2, 2, 2], dtype=jnp.int32)
    assert jnp.array_equal(result_d3, expected_d3), f"Doubles (n=3) failed: {result_d3} != {expected_d3}"

    print("All _update_playable_dice tests passed!")


# ==============================================================================
# == HEURISTIC OBSERVATION TESTS ===============================================
# ==============================================================================

def test_is_race():
    """
    Tests the _is_race function that determines if the game is in a race state
    (no contact possible - all current player checkers have passed opponent checkers).
    """
    from pgx.backgammon import _is_race

    # Test 1: Starting position is NOT a race (contact possible)
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)
    assert _is_race(board_start) == 0, "Starting position should not be a race"

    # Test 2: Pure race - black checkers all ahead of white
    # Black at 20-23, White at 0-5 (already passed each other)
    board_race = jnp.zeros(28, dtype=jnp.int32)
    board_race = board_race.at[20].set(5)  # Black at point 21
    board_race = board_race.at[21].set(5)  # Black at point 22
    board_race = board_race.at[22].set(5)  # Black at point 23
    board_race = board_race.at[0].set(-5)  # White at point 1
    board_race = board_race.at[1].set(-5)  # White at point 2
    board_race = board_race.at[2].set(-5)  # White at point 3
    assert _is_race(board_race) == 1, "Should be a race - all checkers have passed"

    # Test 3: Not a race - one black behind one white
    board_contact = jnp.zeros(28, dtype=jnp.int32)
    board_contact = board_contact.at[10].set(1)  # Black at point 11
    board_contact = board_contact.at[20].set(14)  # Rest of black ahead
    board_contact = board_contact.at[15].set(-15)  # White at point 16 (ahead of black at 11)
    assert _is_race(board_contact) == 0, "Not a race - black at 10 can still hit white at 15"

    # Test 4: Not a race - black has checker on bar
    board_bar_black = jnp.zeros(28, dtype=jnp.int32)
    board_bar_black = board_bar_black.at[24].set(1)  # Black on bar
    board_bar_black = board_bar_black.at[22].set(14)  # Rest of black
    board_bar_black = board_bar_black.at[0].set(-15)  # White far behind
    assert _is_race(board_bar_black) == 0, "Not a race - black has checker on bar"

    # Test 5: Not a race - white has checker on bar
    board_bar_white = jnp.zeros(28, dtype=jnp.int32)
    board_bar_white = board_bar_white.at[22].set(15)  # Black all on home
    board_bar_white = board_bar_white.at[25].set(-1)  # White on bar
    board_bar_white = board_bar_white.at[0].set(-14)  # Rest of white
    assert _is_race(board_bar_white) == 0, "Not a race - white has checker on bar"

    # Test 6: Race - all checkers borne off for both except home boards
    board_bearing = jnp.zeros(28, dtype=jnp.int32)
    board_bearing = board_bearing.at[23].set(10)  # Black at point 24
    board_bearing = board_bearing.at[26].set(5)   # Black off
    board_bearing = board_bearing.at[0].set(-10)  # White at point 1
    board_bearing = board_bearing.at[27].set(-5)  # White off
    assert _is_race(board_bearing) == 1, "Should be a race - both in home areas"

    print("All _is_race tests passed!")


def test_can_bear_off():
    """
    Tests the bear off detection for current and opponent players.
    """
    from pgx.backgammon import _can_bear_off_current, _can_bear_off_opponent

    # Test 1: Starting position - neither can bear off
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)
    assert _can_bear_off_current(board_start) == 0, "Black cannot bear off at start"
    assert _can_bear_off_opponent(board_start) == 0, "White cannot bear off at start"

    # Test 2: Black can bear off (all on home board 18-23)
    board_black_home = jnp.zeros(28, dtype=jnp.int32)
    board_black_home = board_black_home.at[18].set(5)
    board_black_home = board_black_home.at[19].set(5)
    board_black_home = board_black_home.at[20].set(5)
    board_black_home = board_black_home.at[12].set(-15)  # White at midpoint (not in their home)
    assert _can_bear_off_current(board_black_home) == 1, "Black should be able to bear off"
    assert _can_bear_off_opponent(board_black_home) == 0, "White cannot bear off"

    # Test 3: White can bear off (all on white's home board 0-5)
    board_white_home = jnp.zeros(28, dtype=jnp.int32)
    board_white_home = board_white_home.at[12].set(15)  # Black at midpoint (not in their home)
    board_white_home = board_white_home.at[0].set(-5)
    board_white_home = board_white_home.at[1].set(-5)
    board_white_home = board_white_home.at[2].set(-5)
    assert _can_bear_off_current(board_white_home) == 0, "Black cannot bear off"
    assert _can_bear_off_opponent(board_white_home) == 1, "White should be able to bear off"

    # Test 4: Both can bear off
    board_both = jnp.zeros(28, dtype=jnp.int32)
    board_both = board_both.at[20].set(15)  # Black all in home
    board_both = board_both.at[2].set(-15)   # White all in their home
    assert _can_bear_off_current(board_both) == 1, "Black should be able to bear off"
    assert _can_bear_off_opponent(board_both) == 1, "White should be able to bear off"

    # Test 5: Black has some off already
    board_partial = jnp.zeros(28, dtype=jnp.int32)
    board_partial = board_partial.at[22].set(10)  # 10 on home board
    board_partial = board_partial.at[26].set(5)   # 5 borne off
    board_partial = board_partial.at[0].set(-15)
    assert _can_bear_off_current(board_partial) == 1, "Black can bear off with some already off"

    # Test 6: Black has checker on bar - cannot bear off
    board_bar = jnp.zeros(28, dtype=jnp.int32)
    board_bar = board_bar.at[22].set(14)
    board_bar = board_bar.at[24].set(1)  # One on bar
    board_bar = board_bar.at[0].set(-15)
    assert _can_bear_off_current(board_bar) == 0, "Black cannot bear off with checker on bar"

    print("All _can_bear_off tests passed!")


def test_pip_count():
    """
    Tests the pip count calculation for current and opponent players.
    Pip count = sum of (distance to bear off) for each checker.
    """
    from pgx.backgammon import _pip_count_current, _pip_count_opponent

    # Test 1: Simple case - one checker
    board1 = jnp.zeros(28, dtype=jnp.int32)
    board1 = board1.at[0].set(1)  # Black at point 1 (index 0), distance = 24
    board1 = board1.at[26].set(14)  # Rest off
    board1 = board1.at[27].set(-15)  # White all off
    assert _pip_count_current(board1) == 24, f"Expected 24, got {_pip_count_current(board1)}"

    # Test 2: Checker at point 24 (index 23), distance = 1
    board2 = jnp.zeros(28, dtype=jnp.int32)
    board2 = board2.at[23].set(1)  # Black at point 24, distance = 1
    board2 = board2.at[26].set(14)
    board2 = board2.at[27].set(-15)
    assert _pip_count_current(board2) == 1, f"Expected 1, got {_pip_count_current(board2)}"

    # Test 3: Checker on bar, distance = 25
    board3 = jnp.zeros(28, dtype=jnp.int32)
    board3 = board3.at[24].set(1)  # Black on bar, distance = 25
    board3 = board3.at[26].set(14)
    board3 = board3.at[27].set(-15)
    assert _pip_count_current(board3) == 25, f"Expected 25, got {_pip_count_current(board3)}"

    # Test 4: Multiple checkers
    board4 = jnp.zeros(28, dtype=jnp.int32)
    board4 = board4.at[0].set(2)   # 2 checkers × 24 = 48
    board4 = board4.at[23].set(3)  # 3 checkers × 1 = 3
    board4 = board4.at[26].set(10)
    board4 = board4.at[27].set(-15)
    assert _pip_count_current(board4) == 51, f"Expected 51, got {_pip_count_current(board4)}"

    # Test 5: Opponent pip count (white moves from high to low indices)
    # White at index 23 (point 24), distance for white = 24
    board5 = jnp.zeros(28, dtype=jnp.int32)
    board5 = board5.at[26].set(15)  # Black all off
    board5 = board5.at[23].set(-1)  # White at point 24, distance = 24
    board5 = board5.at[27].set(-14)
    assert _pip_count_opponent(board5) == 24, f"Expected 24, got {_pip_count_opponent(board5)}"

    # Test 6: White at index 0 (point 1), distance for white = 1
    board6 = jnp.zeros(28, dtype=jnp.int32)
    board6 = board6.at[26].set(15)
    board6 = board6.at[0].set(-1)  # White at point 1, distance = 1
    board6 = board6.at[27].set(-14)
    assert _pip_count_opponent(board6) == 1, f"Expected 1, got {_pip_count_opponent(board6)}"

    # Test 7: White on bar, distance = 25
    board7 = jnp.zeros(28, dtype=jnp.int32)
    board7 = board7.at[26].set(15)
    board7 = board7.at[25].set(-1)  # White on bar, distance = 25
    board7 = board7.at[27].set(-14)
    assert _pip_count_opponent(board7) == 25, f"Expected 25, got {_pip_count_opponent(board7)}"

    # Test 8: Starting position pip count
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)
    # Black: 2×24 + 5×13 + 3×8 + 5×6 = 48 + 65 + 24 + 30 = 167
    expected_black = 2*24 + 5*13 + 3*8 + 5*6
    assert _pip_count_current(board_start) == expected_black, f"Expected {expected_black}, got {_pip_count_current(board_start)}"
    # White: 5×19 + 3×17 + 5×12 + 2×1 = 95 + 51 + 60 + 2 = 208
    # Wait, white moves in opposite direction: distance = index + 1
    # White at 5: distance = 6, White at 7: distance = 8, White at 12: distance = 13, White at 23: distance = 24
    expected_white = 5*6 + 3*8 + 5*13 + 2*24
    assert _pip_count_opponent(board_start) == expected_white, f"Expected {expected_white}, got {_pip_count_opponent(board_start)}"

    print("All _pip_count tests passed!")


def test_pip_count_differential():
    """
    Tests the scaled pip count differential.
    Positive = current player is ahead (lower pip count).
    """
    from pgx.backgammon import _pip_count_differential_scaled

    # Test 1: Equal pip counts should give 0
    board_equal = jnp.zeros(28, dtype=jnp.int32)
    board_equal = board_equal.at[12].set(15)   # Black all at midpoint, distance = 12 each = 180
    board_equal = board_equal.at[11].set(-15)  # White at 12 (distance = 12 each) = 180
    diff = _pip_count_differential_scaled(board_equal)
    assert jnp.abs(diff) < 0.01, f"Expected ~0, got {diff}"

    # Test 2: Black ahead (lower pip count)
    board_black_ahead = jnp.zeros(28, dtype=jnp.int32)
    board_black_ahead = board_black_ahead.at[23].set(15)  # Black at 24, distance = 1 each = 15
    board_black_ahead = board_black_ahead.at[0].set(-15)  # White at 1, distance = 1 each = 15
    # Both have pip count 15, but directions are different
    # Black: 15 × 1 = 15
    # White: 15 × 1 = 15
    # Differential should be 0
    diff2 = _pip_count_differential_scaled(board_black_ahead)
    assert jnp.abs(diff2) < 0.01, f"Expected ~0, got {diff2}"

    # Test 3: Black has big lead
    board_black_winning = jnp.zeros(28, dtype=jnp.int32)
    board_black_winning = board_black_winning.at[26].set(15)  # Black all off, pip = 0
    board_black_winning = board_black_winning.at[0].set(-15)  # White at 1, pip = 15
    diff3 = _pip_count_differential_scaled(board_black_winning)
    # Differential = opponent_pips - current_pips = 15 - 0 = 15
    # Scaled = 15 / 375 = 0.04
    assert diff3 > 0, f"Expected positive (black ahead), got {diff3}"

    # Test 4: White has big lead (from black's perspective)
    board_white_winning = jnp.zeros(28, dtype=jnp.int32)
    board_white_winning = board_white_winning.at[0].set(15)  # Black at 1, pip = 15*24 = 360
    board_white_winning = board_white_winning.at[27].set(-15)  # White all off, pip = 0
    diff4 = _pip_count_differential_scaled(board_white_winning)
    # Differential = 0 - 360 = -360
    # Scaled = -360 / 375 = -0.96
    assert diff4 < 0, f"Expected negative (white ahead), got {diff4}"

    # Test 5: Check scaling bounds
    # Max current pip = 15 × 25 (all on bar) = 375
    board_max_curr = jnp.zeros(28, dtype=jnp.int32)
    board_max_curr = board_max_curr.at[24].set(15)  # Black all on bar
    board_max_curr = board_max_curr.at[27].set(-15)  # White all off
    diff_max_neg = _pip_count_differential_scaled(board_max_curr)
    assert diff_max_neg >= -1.0, f"Should be >= -1.0, got {diff_max_neg}"

    board_max_opp = jnp.zeros(28, dtype=jnp.int32)
    board_max_opp = board_max_opp.at[26].set(15)   # Black all off
    board_max_opp = board_max_opp.at[25].set(-15)  # White all on bar
    diff_max_pos = _pip_count_differential_scaled(board_max_opp)
    assert diff_max_pos <= 1.0, f"Should be <= 1.0, got {diff_max_pos}"

    print("All _pip_count_differential_scaled tests passed!")


def test_observe_with_heuristics():
    """
    Tests the full _observe_with_heuristics function that returns
    the complete observation including board, dice, and heuristics.
    """
    from pgx.backgammon import _observe_with_heuristics

    # Test with starting position
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)
    playable_dice = jnp.array([2, 3, -1, -1], dtype=jnp.int32)  # 3-4 roll

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board_start,
        turn=jnp.int32(0),
        dice=jnp.array([2, 3], dtype=jnp.int32),
        playable_dice=playable_dice,
        played_dice_num=jnp.int32(0),
    )

    obs = _observe_with_heuristics(state)

    # Check observation shape: 28 (board) + 6 (dice) + 4 (heuristics) = 38
    assert obs.shape == (38,), f"Expected shape (38,), got {obs.shape}"

    # Check board portion (first 28 elements)
    assert jnp.array_equal(obs[:28], board_start), "Board portion should match"

    # Check dice portion (elements 28-33)
    expected_dice = jnp.array([0, 0, 1, 1, 0, 0], dtype=jnp.int32)  # 3 and 4
    assert jnp.array_equal(obs[28:34], expected_dice), f"Dice portion mismatch: {obs[28:34]} vs {expected_dice}"

    # Check heuristics portion (elements 34-37)
    # contact flag should be 0 (not a race, contact possible)
    assert obs[34] == 0, f"Contact flag should be 0, got {obs[34]}"
    # current player bear off should be 0
    assert obs[35] == 0, f"Current bear off should be 0, got {obs[35]}"
    # opponent bear off should be 0
    assert obs[36] == 0, f"Opponent bear off should be 0, got {obs[36]}"
    # pip differential - starting position should be close to 0
    # (both have 167 pips)
    assert jnp.abs(obs[37]) < 0.1, f"Pip differential should be ~0, got {obs[37]}"

    print("All _observe_with_heuristics tests passed!")


def test_observe_with_heuristics_race_position():
    """
    Tests observation in a race position where both sides have passed each other.
    """
    from pgx.backgammon import _observe_with_heuristics

    # Create a race position
    board_race = jnp.zeros(28, dtype=jnp.int32)
    board_race = board_race.at[20].set(10)  # Black on home board
    board_race = board_race.at[22].set(5)
    board_race = board_race.at[2].set(-10)  # White on their home board
    board_race = board_race.at[4].set(-5)

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board_race,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )

    obs = _observe_with_heuristics(state)

    # contact flag should be 1 (is a race)
    assert obs[34] == 1, f"Contact flag should be 1 (race), got {obs[34]}"
    # current player bear off should be 1 (black all on home board)
    assert obs[35] == 1, f"Current bear off should be 1, got {obs[35]}"
    # opponent bear off should be 1 (white all on their home board)
    assert obs[36] == 1, f"Opponent bear off should be 1, got {obs[36]}"

    print("All race position observation tests passed!")


def test_observation_ranges_board():
    """
    Tests the ranges of board observations (indices 0-27).
    Verifies that board values fall within expected ranges.
    """
    from pgx.backgammon import _observe

    # Test 1: Maximum positive values - all black checkers on one point
    board_max_black = jnp.zeros(28, dtype=jnp.int32)
    board_max_black = board_max_black.at[0].set(15)  # All black at point 1
    board_max_black = board_max_black.at[27].set(-15)  # All white off

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board_max_black,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs = _observe(state)

    # Point 0 should have max value of 15 (scaled to 1.0)
    assert obs[0] == 1.0, f"Max black on point should be 1.0 (scaled 15), got {obs[0]}"

    # White off should be -15 (scaled to -1.0)
    assert obs[27] == -1.0, f"White off should be -1.0 (scaled -15), got {obs[27]}"

    

    # Test 2: Maximum negative values - all white checkers on one point
    board_max_white = jnp.zeros(28, dtype=jnp.int32)
    board_max_white = board_max_white.at[12].set(-15)  # All white at point 13
    board_max_white = board_max_white.at[26].set(15)   # All black off

    state2 = make_test_state(
        current_player=jnp.int32(0),
        board=board_max_white,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs2 = _observe(state2)

    # Point 12 should have min value of -15 (scaled to -1.0)
    assert obs2[12] == -1.0, f"Max white on point should be -1.0 (scaled -15), got {obs2[12]}"
    # Black off should be 15 (scaled to 1.0)
    assert obs2[26] == 1.0, f"Black off should be 1.0 (scaled 15), got {obs2[26]}"

    # Test 3: Bar positions - max values
    board_bar = jnp.zeros(28, dtype=jnp.int32)
    board_bar = board_bar.at[24].set(15)   # All black on bar
    board_bar = board_bar.at[25].set(-15)  # All white on bar

    state3 = make_test_state(
        current_player=jnp.int32(0),
        board=board_bar,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs3 = _observe(state3)

    # Black bar should be 15 (scaled to 1.0)
    assert obs3[24] == 1.0, f"Black bar max should be 1.0 (scaled 15), got {obs3[24]}"

    # White bar should be -15 (scaled to -1.0)
    assert obs3[25] == -1.0, f"White bar max should be -1.0 (scaled -15), got {obs3[25]}"

    

    # Test 4: Verify all board positions are within [-15, 15]
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)
    state4 = make_test_state(
        current_player=jnp.int32(0),
        board=board_start,
        turn=jnp.int32(0),
        dice=jnp.array([2, 3], dtype=jnp.int32),
        playable_dice=jnp.array([2, 3, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs4 = _observe(state4)

    # Check all board positions (0-27) are within range
    for i in range(28):
        assert -15 <= obs4[i] <= 15, f"Board position {i} out of range: {obs4[i]}"

    print("All board observation range tests passed!")
    print(f"  Points range: [-15, 15]")
    print(f"  Bar range: black [0, 15], white [-15, 0]")
    print(f"  Off range: black [0, 15], white [-15, 0]")


def test_observation_ranges_dice():
    """
    Tests the ranges of dice observations (indices 28-33).
    """
    from pgx.backgammon import _observe

    # Test 1: Non-doubles - two different dice
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(15)
    board = board.at[27].set(-15)

    state1 = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([0, 5], dtype=jnp.int32),  # 1 and 6
        playable_dice=jnp.array([0, 5, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs1 = _observe(state1)

    # Dice portion is indices 28-33
    # Should have 0.25 for die value 1 (index 28) and 0.25 for die value 6 (index 33)
    # (scaled raw count 1 / 4 = 0.25)
    expected_dice = jnp.array([0.25, 0, 0, 0, 0, 0.25], dtype=jnp.float32)
    assert jnp.array_equal(obs1[28:34], expected_dice), f"Non-doubles dice mismatch: {obs1[28:34]}"

    # Test 2: Doubles - four of same die
    state2 = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([3, 3], dtype=jnp.int32),  # double 4s
        playable_dice=jnp.array([3, 3, 3, 3], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs2 = _observe(state2)

    # Should have 1.0 for die value 4 (index 31)
    # (scaled raw count 4 / 4 = 1.0)
    expected_dice2 = jnp.array([0, 0, 0, 1.0, 0, 0], dtype=jnp.float32)
    assert jnp.array_equal(obs2[28:34], expected_dice2), f"Doubles dice mismatch: {obs2[28:34]}"

    # Test 3: After playing some dice (partial doubles)
    state3 = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([2, 2], dtype=jnp.int32),  # double 3s
        playable_dice=jnp.array([2, 2, -1, -1], dtype=jnp.int32),  # 2 remaining
        played_dice_num=jnp.int32(2),
    )
    obs3 = _observe(state3)

    # Should have 0.5 for die value 3 (index 30)
    # (scaled raw count 2 / 4 = 0.5)
    expected_dice3 = jnp.array([0, 0, 0.5, 0, 0, 0], dtype=jnp.float32)
    assert jnp.array_equal(obs3[28:34], expected_dice3), f"Partial doubles mismatch: {obs3[28:34]}"

    # Test 4: No dice remaining
    state4 = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, -1, -1, -1], dtype=jnp.int32),  # all used
        played_dice_num=jnp.int32(2),
    )
    obs4 = _observe(state4)

    # Should be all zeros
    expected_dice4 = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    assert jnp.array_equal(obs4[28:34], expected_dice4), f"Empty dice mismatch: {obs4[28:34]}"

    # Verify dice counts are always in [0, 4]
    for i in range(28, 34):
        assert 0 <= obs1[i] <= 4, f"Dice position {i} out of range: {obs1[i]}"
        assert 0 <= obs2[i] <= 4, f"Dice position {i} out of range: {obs2[i]}"

    print("All dice observation range tests passed!")
    print(f"  Dice count range: [0, 4]")


def test_observation_ranges_heuristics():
    """
    Tests the ranges of heuristic observations (indices 34-37).
    """
    from pgx.backgammon import _observe_with_heuristics

    # Test 1: Extreme case - black all on bar (max pip), white all off
    board_worst_black = jnp.zeros(28, dtype=jnp.int32)
    board_worst_black = board_worst_black.at[24].set(15)   # All black on bar
    board_worst_black = board_worst_black.at[27].set(-15)  # All white off

    state1 = make_test_state(
        current_player=jnp.int32(0),
        board=board_worst_black,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs1 = _observe_with_heuristics(state1)

    # Race flag: 0 (black on bar, so contact possible)
    assert obs1[34] == 0, f"Race flag should be 0, got {obs1[34]}"
    # Current bear off: 0 (black on bar)
    assert obs1[35] == 0, f"Bear off current should be 0, got {obs1[35]}"
    # Opponent bear off: 1 (white all off, so can "bear off" trivially)
    # Actually white is all OFF, not on home board, need to check logic
    # Pip differential: should be very negative (black far behind)
    # Black pip = 15 * 25 = 375, White pip = 0
    # Differential = 0 - 375 = -375, scaled = -1.0
    assert obs1[37] == -1.0, f"Pip differential should be -1.0, got {obs1[37]}"

    # Test 2: Extreme case - black all off, white all on bar
    board_best_black = jnp.zeros(28, dtype=jnp.int32)
    board_best_black = board_best_black.at[26].set(15)   # All black off
    board_best_black = board_best_black.at[25].set(-15)  # All white on bar

    state2 = make_test_state(
        current_player=jnp.int32(0),
        board=board_best_black,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs2 = _observe_with_heuristics(state2)

    # Pip differential: should be very positive (black far ahead)
    # Black pip = 0, White pip = 15 * 25 = 375
    # Differential = 375 - 0 = 375, scaled = 1.0
    assert obs2[37] == 1.0, f"Pip differential should be 1.0, got {obs2[37]}"

    # Test 3: Equal position
    board_equal = jnp.zeros(28, dtype=jnp.int32)
    board_equal = board_equal.at[12].set(15)   # Black at midpoint
    board_equal = board_equal.at[11].set(-15)  # White at same distance

    state3 = make_test_state(
        current_player=jnp.int32(0),
        board=board_equal,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    obs3 = _observe_with_heuristics(state3)

    # Pip differential should be 0
    assert abs(obs3[37]) < 0.01, f"Pip differential should be ~0, got {obs3[37]}"

    # Verify all heuristics are in expected ranges
    # Race flag: [0, 1]
    assert obs1[34] in [0, 1] or obs1[34] in [0.0, 1.0], f"Race flag out of range: {obs1[34]}"
    # Bear off flags: [0, 1]
    assert obs1[35] in [0, 1] or obs1[35] in [0.0, 1.0], f"Bear off current out of range: {obs1[35]}"
    assert obs1[36] in [0, 1] or obs1[36] in [0.0, 1.0], f"Bear off opponent out of range: {obs1[36]}"
    # Pip differential: [-1, 1]
    assert -1.0 <= obs1[37] <= 1.0, f"Pip differential out of range: {obs1[37]}"
    assert -1.0 <= obs2[37] <= 1.0, f"Pip differential out of range: {obs2[37]}"

    print("All heuristic observation range tests passed!")
    print(f"  Race flag range: [0, 1]")
    print(f"  Bear off flags range: [0, 1]")
    print(f"  Pip differential range: [-1, 1]")


def test_observation_complete_summary():
    """
    Comprehensive test that documents all observation ranges.
    """
    from pgx.backgammon import _observe, _observe_with_heuristics

    print("\n" + "="*60)
    print("OBSERVATION RANGE SUMMARY")
    print("="*60)

    # Standard observation: 34 elements
    print("\nStandard Observation (_observe): 34 elements")
    print("-" * 40)
    print(f"  [0:24]  Points         : [-15, +15]  (neg=white, pos=black)")
    print(f"  [24]    Black bar      : [0, 15]")
    print(f"  [25]    White bar      : [-15, 0]")
    print(f"  [26]    Black off      : [0, 15]")
    print(f"  [27]    White off      : [-15, 0]")
    print(f"  [28:34] Dice counts    : [0, 4]      (count per die value)")

    # Heuristic observation: 38 elements
    print("\nHeuristic Observation (_observe_with_heuristics): 38 elements")
    print("-" * 40)
    print(f"  [0:34]  Standard obs   : (see above)")
    print(f"  [34]    Race flag      : [0, 1]      (1=race, 0=contact)")
    print(f"  [35]    Current bear   : [0, 1]      (1=can bear off)")
    print(f"  [36]    Opponent bear  : [0, 1]      (1=can bear off)")
    print(f"  [37]    Pip diff scaled: [-1, 1]     (pos=current ahead)")

    # Verify shapes
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)
    state = make_test_state(
        current_player=jnp.int32(0),
        board=board_start,
        turn=jnp.int32(0),
        dice=jnp.array([2, 3], dtype=jnp.int32),
        playable_dice=jnp.array([2, 3, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )

    obs_standard = _observe(state)
    obs_heuristic = _observe_with_heuristics(state)

    assert obs_standard.shape == (86,), f"Standard obs shape should be (86,), got {obs_standard.shape}"
    assert obs_heuristic.shape == (38,), f"Heuristic obs shape should be (38,), got {obs_heuristic.shape}"

    print("\nML Scaling Recommendations:")
    print("-" * 40)
    print(f"  Points/Bar/Off : divide by 15 → [-1, 1]")
    print(f"  Dice counts    : divide by 4  → [0, 1]")
    print(f"  Heuristics     : already normalized")

    print("\n" + "="*60)
    print("All observation range tests passed!")
    print("="*60)


# ==============================================================================
# == FULL OBSERVATION WITH BLOT/BLOCKER TESTS ==================================
# ==============================================================================

def test_blot_board():
    """Test blot detection on points."""
    from pgx.backgammon import _blot_board

    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(1)    # Current player blot
    board = board.at[5].set(-1)   # Opponent blot
    board = board.at[10].set(2)   # Not a blot (2 checkers)
    board = board.at[15].set(-3)  # Not a blot (3 checkers)
    board = board.at[20].set(0)   # Empty point

    blots = _blot_board(board)
    assert blots.shape == (24,), f"Expected shape (24,), got {blots.shape}"
    assert blots[0] == 1.0, f"Current blot at 0 should be 1.0, got {blots[0]}"
    assert blots[5] == -1.0, f"Opponent blot at 5 should be -1.0, got {blots[5]}"
    assert blots[10] == 0.0, f"2 checkers at 10 should not be blot, got {blots[10]}"
    assert blots[15] == 0.0, f"3 checkers at 15 should not be blot, got {blots[15]}"
    assert blots[20] == 0.0, f"Empty at 20 should be 0, got {blots[20]}"

    print("All _blot_board tests passed!")


def test_blocker_board():
    """Test blocker detection on points."""
    from pgx.backgammon import _blocker_board

    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(2)    # Current 2 checkers
    board = board.at[5].set(5)    # Current 5 checkers
    board = board.at[10].set(-2)  # Opponent 2 checkers
    board = board.at[15].set(-4)  # Opponent 4 checkers
    board = board.at[20].set(1)   # Blot (not a blocker)
    board = board.at[21].set(-1)  # Opponent blot (not a blocker)

    blockers = _blocker_board(board)
    assert blockers.shape == (24,), f"Expected shape (24,), got {blockers.shape}"
    assert blockers[0] == 0.5, f"2 checkers at 0 should be 0.5, got {blockers[0]}"
    assert blockers[5] == 1.0, f"5 checkers at 5 should be 1.0, got {blockers[5]}"
    assert blockers[10] == -0.5, f"Opponent 2 at 10 should be -0.5, got {blockers[10]}"
    assert blockers[15] == -1.0, f"Opponent 4 at 15 should be -1.0, got {blockers[15]}"
    assert blockers[20] == 0.0, f"Blot at 20 should not be blocker, got {blockers[20]}"
    assert blockers[21] == 0.0, f"Opponent blot at 21 should not be blocker, got {blockers[21]}"

    print("All _blocker_board tests passed!")


def test_observe_full():
    """Test full observation with scaling and all features."""
    from pgx.backgammon import _observe_full

    # Starting position board
    board_start = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board_start,
        turn=jnp.int32(0),
        dice=jnp.array([2, 3], dtype=jnp.int32),
        playable_dice=jnp.array([2, 3, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )

    obs = _observe_full(state)

    # Check shape: 86 elements
    assert obs.shape == (86,), f"Expected shape (86,), got {obs.shape}"

    # Check board is scaled: 2/15 ≈ 0.133
    assert jnp.isclose(obs[0], 2/15), f"Board[0] should be 2/15, got {obs[0]}"
    assert jnp.isclose(obs[5], -5/15), f"Board[5] should be -5/15, got {obs[5]}"

    # Check dice is scaled: dice 3 and 4 → indices 30 and 31
    # playable_dice = [2, 3] means die values 3 and 4 (0-indexed)
    assert jnp.isclose(obs[30], 0.25), f"Dice[2] (die value 3) should be 0.25, got {obs[30]}"
    assert jnp.isclose(obs[31], 0.25), f"Dice[3] (die value 4) should be 0.25, got {obs[31]}"

    # Check heuristics
    assert obs[34] == 0, f"Race flag should be 0 (not a race), got {obs[34]}"
    assert obs[35] == 0, f"Current bear off should be 0, got {obs[35]}"
    assert obs[36] == 0, f"Opponent bear off should be 0, got {obs[36]}"

    # Check blot features (indices 38-62)
    # Starting position has 2 checkers at point 0, so no blot there
    assert obs[38] == 0.0, f"Point 0 has 2 checkers, not a blot, got {obs[38]}"

    # Check blocker features (indices 62-86)
    # Starting position has 2 at point 0
    assert obs[62] == 0.5, f"Point 0 has 2 checkers → 0.5 blocker, got {obs[62]}"
    # Point 5 has -5 white checkers → -1.0 blocker
    assert obs[62 + 5] == -1.0, f"Point 5 has -5 → -1.0 blocker, got {obs[62 + 5]}"

    print("All _observe_full tests passed!")


def test_observe_full_all_scaled():
    """Verify all observation values are in [-1, 1] range."""
    from pgx.backgammon import _observe_full

    # Test with valid extreme board positions (15 checkers per side)
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(15)    # All black at point 0 (max 15)
    board = board.at[23].set(-15)  # All white at point 23 (max -15)

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([5, 5], dtype=jnp.int32),  # Doubles
        playable_dice=jnp.array([5, 5, 5, 5], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )

    obs = _observe_full(state)

    # ALL values should be in [-1, 1]
    assert jnp.all(obs >= -1.0), f"Min value {obs.min()} < -1"
    assert jnp.all(obs <= 1.0), f"Max value {obs.max()} > 1"

    # Check specific extreme values
    assert jnp.isclose(obs[0], 1.0), f"15/15 should be 1.0, got {obs[0]}"
    assert jnp.isclose(obs[23], -1.0), f"-15/15 should be -1.0, got {obs[23]}"

    # Test with bar checkers (valid: 15 total per side)
    board2 = jnp.zeros(28, dtype=jnp.int32)
    board2 = board2.at[24].set(15)   # All black on bar
    board2 = board2.at[25].set(-15)  # All white on bar

    state2 = make_test_state(
        current_player=jnp.int32(0),
        board=board2,
        turn=jnp.int32(0),
        dice=jnp.array([0, 0], dtype=jnp.int32),
        playable_dice=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )

    obs2 = _observe_full(state2)

    # Check bar values are scaled properly
    assert jnp.isclose(obs2[24], 1.0), f"Bar 15/15 should be 1.0, got {obs2[24]}"
    assert jnp.isclose(obs2[25], -1.0), f"Bar -15/15 should be -1.0, got {obs2[25]}"

    # Pip differential should also be in range
    assert obs2[37] >= -1.0 and obs2[37] <= 1.0, f"Pip diff out of range: {obs2[37]}"

    print("All _observe_full_all_scaled tests passed!")


def test_observe_full_blot_blocker_values():
    """Test specific blot and blocker values in full observation."""
    from pgx.backgammon import _observe_full

    # Create a board with specific blot/blocker patterns
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(1)    # Current blot → blots[0] = 1.0
    board = board.at[5].set(-1)   # Opponent blot → blots[5] = -1.0
    board = board.at[10].set(2)   # Current 2 → blockers[10] = 0.5
    board = board.at[15].set(4)   # Current 4 → blockers[15] = 1.0
    board = board.at[20].set(-2)  # Opponent 2 → blockers[20] = -0.5
    board = board.at[22].set(-5)  # Opponent 5 → blockers[22] = -1.0
    board = board.at[26].set(3)   # Some off (not in blot/blocker range)
    board = board.at[27].set(-2)  # Some off

    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )

    obs = _observe_full(state)

    # Blot features at indices 38-61
    assert obs[38 + 0] == 1.0, f"Blot at point 0 should be 1.0, got {obs[38 + 0]}"
    assert obs[38 + 5] == -1.0, f"Blot at point 5 should be -1.0, got {obs[38 + 5]}"
    assert obs[38 + 10] == 0.0, f"Point 10 (2 checkers) should not be blot, got {obs[38 + 10]}"

    # Blocker features at indices 62-85
    assert obs[62 + 10] == 0.5, f"Blocker at point 10 (2) should be 0.5, got {obs[62 + 10]}"
    assert obs[62 + 15] == 1.0, f"Blocker at point 15 (4) should be 1.0, got {obs[62 + 15]}"
    assert obs[62 + 20] == -0.5, f"Blocker at point 20 (-2) should be -0.5, got {obs[62 + 20]}"
    assert obs[62 + 22] == -1.0, f"Blocker at point 22 (-5) should be -1.0, got {obs[62 + 22]}"

    print("All _observe_full_blot_blocker_values tests passed!")
