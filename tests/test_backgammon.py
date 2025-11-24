from functools import partial
import jax
import jax.numpy as jnp
from pgx.backgammon import (
    State,
    _flip_board,
    _calc_src,
    _calc_tgt,
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
    turn_to_str
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
    _legal_action_mask as _legal_action_mask_v1,
    _change_turn as _change_turn_v1,
    _init as _init_v1,
    _no_winning_step as _no_winning_step_v1,
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
_calc_src = jax.jit(_calc_src)
_calc_tgt = jax.jit(_calc_tgt)
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
):
    return State(
        current_player=current_player,
        _board=board,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
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
    state = _change_turn_v1(state, jax.random.PRNGKey(0))
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
    state = _change_turn_v1(state, jax.random.PRNGKey(0))
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
    expected_obs = jnp.concatenate(
        (board, jnp.array([1, 1, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([1, 1, 1, 1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 4, 0, 0, 0, 0])), axis=None
    )
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
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 1, 0, 0, 0, 0])), axis=None
    )
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
    expected_obs = jnp.concatenate(
        (board, jnp.array([0, 1, 0, 0, 0, 0])), axis=None
    )
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


def test_calc_src():
    assert _calc_src(1) == 24
    assert _calc_src(2) == 0


def test_calc_tgt():
    assert _calc_tgt(24, 1) == 0  # bar to board (die is transformed from 0~5 -> 1~ 6)
    assert _calc_tgt(6, 2) == 8  # board to board
    assert _calc_tgt(23, 6) == 26  # to off


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
    assert state.is_stochastic  # type: ignore
    
    # After a stochastic step with doubles, it should no longer be stochastic
    stochastic_action = 0  # Using double 1's (action 0)
    new_state: State = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    assert not new_state.is_stochastic  # type: ignore
    
    # With doubles, player should be able to make 4 moves before state becomes stochastic again
    for _ in range(4):
        # State should remain non-stochastic during moves
        assert not new_state.is_stochastic  # type: ignore
        
        # Make a move
        legal_action = jnp.where(new_state.legal_action_mask)[0][0]
        new_state = env.step(new_state, legal_action, jax.random.PRNGKey(1))  # type: ignore
    
    # After all 4 moves are made, state should be stochastic again
    assert new_state.is_stochastic  # type: ignore


def test_stochastic_actions():
    """Test getting available stochastic actions and their probabilities."""
    # For regular mode, all 21 dice combinations should be possible
    assert len(env.stochastic_action_probs) == 21
    
    # Test that probabilities sum to 1
    assert jnp.isclose(jnp.sum(env.stochastic_action_probs), 1.0)
    
    # Test simple doubles mode
    env_simple = Backgammon(simple_doubles=True)
    assert len(env_simple.stochastic_action_probs) == 21
    
    # In simple doubles mode, only the first 6 actions (doubles) have non-zero probability
    assert jnp.all(env_simple.stochastic_action_probs[6:] == 0)
    assert jnp.isclose(jnp.sum(env_simple.stochastic_action_probs), 1.0)


def test_stochastic_step():
    """Test applying a stochastic action to set dice."""
    state: State = env.init(jax.random.PRNGKey(0))  # type: ignore
    
    # Apply stochastic action 0 (double 1's)
    stochastic_action = 0
    new_state: State = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    
    # Check that dice are set correctly
    assert jnp.array_equal(new_state._dice, jnp.array([0, 0]))  # type: ignore
    
    # Check that state is no longer stochastic
    assert not new_state.is_stochastic  # type: ignore
    
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
    assert not new_state.is_stochastic  # type: ignore
    

def test_stochastic_game_simulation():
    """Test simulating a game with predefined dice rolls."""
    # Initialize game
    state: State = env.init(jax.random.PRNGKey(42))  # type: ignore
    
    # Test roll: 1,6 (action index 10)
    assert state.is_stochastic  # type: ignore
    state = env.stochastic_step(state, jnp.array(10))  # type: ignore
    assert not state.is_stochastic  # type: ignore
    assert jnp.array_equal(state._dice, jnp.array([0, 5]))  # type: ignore
    
    # Make a move using the first die
    legal_action = jnp.where(state.legal_action_mask)[0][0]
    state = env.step(state, legal_action, jax.random.PRNGKey(0))  # type: ignore
    
    # Make another move if the turn hasn't changed
    for x in range(20):
        if state.is_stochastic:  # type: ignore
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