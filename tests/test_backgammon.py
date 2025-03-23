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
    Backgammon
)
import os
from pgx._src.api_test import (
    _validate_state,
    _validate_init_reward,
    _validate_current_player,
    _validate_legal_actions,
)

seed = 1701
rng = jax.random.PRNGKey(seed)
env = Backgammon()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
_no_winning_step = jax.jit(_no_winning_step)
_calc_src = jax.jit(_calc_src)
_calc_tgt = jax.jit(_calc_tgt)
_calc_win_score = jax.jit(_calc_win_score)
_change_turn = jax.jit(_change_turn)
_is_action_legal = jax.jit(_is_action_legal)
_is_all_on_home_board = jax.jit(_is_all_on_home_board)
_is_open = jax.jit(_is_open)
_legal_action_mask = jax.jit(_legal_action_mask)
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
    print(_flip_board, test_board)
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
    state = init(rng)
    _turn = state._turn
    state = _change_turn(state, jax.random.PRNGKey(0))
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
    print(state._board, board)
    assert state._turn == jnp.int32(1)  # Turn changed
    assert (state._board == board).all()  # Flipped.


def test_no_op():
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    state = step(state, 0, jax.random.PRNGKey(0))  # execute no-op action
    assert state._turn == jnp.int32(0)  # Turn changes after no-op.


def test_step():
    # 白
    board: jnp.ndarray = make_test_boad()
    board = _flip_board(board)  # Flipped
    legal_action_mask = _legal_action_mask(
        board, jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 1
    ].set(
        True
    )  # 24(bar)->1
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # Test legal action

    # White plays die=2 24(bar)->1
    state = step(state=state, action=(1) * 6 + 1, key=jax.random.PRNGKey(0))
    assert (
            state._playable_dice == jnp.array([0, -1, -1, -1], dtype=jnp.int32)
    ).all()  # Is playable dice updated correctly?
    assert state._played_dice_num == 1  # played dice increased?
    assert state._turn == 1  # turn is not changed?
    assert state._board.at[1].get() == 4 and state._board.at[24].get() == 3
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (1) + 0
    ].set(
        True
    )  # 24(bar)->0
    assert (expected_legal_action_mask == state.legal_action_mask).all()  # test legal action
    # White plays die=1 24(off)->0
    state = step(state=state, action=(1) * 6 + 0, key=jax.random.PRNGKey(0))
    assert state._played_dice_num == 0
    assert state._turn == 0  # turn changed to black?
    assert state._board.at[23].get() == -1 and state._board.at[25].get() == -2
    
    # black
    board: jnp.ndarray = make_test_boad()
    legal_action_mask = _legal_action_mask(
        board, jnp.array([4, 5, -1, -1], dtype=jnp.int32)
    )
    state = make_test_state(
        current_player=jnp.int32(0),
        board=board,
        turn=jnp.int32(0),
        dice=jnp.array([4, 5], dtype=jnp.int32),
        playable_dice=jnp.array([4, 5, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
        legal_action_mask=legal_action_mask,
    )
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(
        True
    )  # 19 -> off
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 4
    ].set(
        True
    )  # 19 -> off
    print(jnp.where(state.legal_action_mask==1)[0], jnp.where(expected_legal_action_mask==1)[0])
    assert (expected_legal_action_mask == state.legal_action_mask).all()


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
    assert (observe(state, jnp.int32(1)) == expected_obs).all()

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
    assert (observe(state, jnp.int32(1)) == expected_obs).all()

    # current_player = black, playabl_dice = (2)
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
    assert (observe(state, jnp.int32(1)) == expected_obs).all()

    state = make_test_state(
        current_player=jnp.int32(1),
        board=board,
        turn=jnp.int32(-1),
        dice=jnp.array([0, 1], dtype=jnp.int32),
        playable_dice=jnp.array([-1, 1, -1, -1], dtype=jnp.int32),
        played_dice_num=jnp.int32(0),
    )
    expected_obs = jnp.concatenate(
        (1 * board, jnp.array([0, 0, 0, 0, 0, 0])), axis=None
    )
    assert (observe(state, jnp.int32(0)) == expected_obs).all()


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
    turn = jnp.int32(-1)
    # Black
    assert _rear_distance(board) == 5
    # White
    board = _flip_board(board)
    assert _rear_distance(board) == 23


def test_distance_to_goal():
    board = make_test_boad()
    # Black
    turn = jnp.int32(-1)
    src = 23
    assert _distance_to_goal(src) == 1
    src = 10
    assert _distance_to_goal(src) == 14
    # Teat at the src where rear_distance is same
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
    print(board)
    assert (
        board.at[1].get() == 2
        and board.at[3].get() == 1
        and board.at[25].get() == -1
    )


def test_legal_action():
    board = make_test_boad()
    # black
    playable_dice = jnp.array([3, 2, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 3
    ].set(
        True
    )  # 19->23
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (20 + 2) + 2
    ].set(
        True
    )  # 20->23
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (20 + 2) + 3
    ].set(
        True
    )  # 20->off
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (21 + 2) + 2
    ].set(
        True
    )  # 21->off
    legal_action_mask = _legal_action_mask(board, playable_dice)
    print(jnp.where(legal_action_mask != 0)[0])
    print(jnp.where(expected_legal_action_mask != 0)[0])
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(6 * 26, dtype=jnp.bool_)
    expected_legal_action_mask = expected_legal_action_mask.at[
        6 * (19 + 2) + 5
    ].set(True)
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    # white
    board = _flip_board(board)
    playable_dice = jnp.array([4, 1, -1, -1], dtype=jnp.int32)
    expected_legal_action_mask: jnp.ndarray = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )
    expected_legal_action_mask = expected_legal_action_mask.at[6 * 1 + 1].set(
        True
    )
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()

    playable_dice = jnp.array([4, 4, 4, 4], dtype=jnp.int32)
    expected_legal_action_mask = jnp.zeros(
        6 * 26, dtype=jnp.bool_
    )  # dance
    expected_legal_action_mask = expected_legal_action_mask.at[0:6].set(
        True
    )  # only no-op
    legal_action_mask = _legal_action_mask(board, playable_dice)
    assert (expected_legal_action_mask == legal_action_mask).all()


def test_calc_win_score():
    # backgammon win by black
    back_gammon_board = jnp.zeros(28, dtype=jnp.int32)
    back_gammon_board = back_gammon_board.at[26].set(15)
    back_gammon_board = back_gammon_board.at[23].set(-15)  # black on home board
    print(_calc_win_score(back_gammon_board))
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


def test_black_off():
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(15)
    playable_dice = jnp.array([3, 2, -1, -1])
    legal_action_mask = _legal_action_mask(board, playable_dice)
    print("3, 2", jnp.where(legal_action_mask != 0)[0])
    playable_dice = jnp.array([1, 1, -1, -1])
    legal_action_mask = _legal_action_mask(board, playable_dice)
    print("1, 1", jnp.where(legal_action_mask != 0)[0])

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
    env = Backgammon()
    state: State = env.init(jax.random.PRNGKey(0))  # type: ignore
    
    # New game state should be stochastic (needs dice)
    assert state.is_stochastic  # type: ignore
    
    # After a stochastic step, it should no longer be stochastic
    stochastic_action = 0  # Using double 1's (action 0)
    new_state: State = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    assert not new_state.is_stochastic  # type: ignore
    
    # After a regular move and turn change, it should be stochastic again
    legal_action = jnp.where(new_state.legal_action_mask)[0][0]
    moved_state: State = env.step(new_state, legal_action, jax.random.PRNGKey(1))  # type: ignore
    
    # If the move ended the turn, it should be stochastic again
    # We need to check if the turn actually changed
    if moved_state._turn != new_state._turn:  # type: ignore
        assert moved_state.is_stochastic  # type: ignore


def test_stochastic_actions():
    """Test getting available stochastic actions and their probabilities."""
    env = Backgammon()
    state: State = env.init(jax.random.PRNGKey(0))  # type: ignore
    
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
    env = Backgammon()
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
    env = Backgammon(simple_doubles=True)
    state: State = env.init(jax.random.PRNGKey(0))  # type: ignore
    
    # In simple doubles mode, only double actions (0-5) should work
    
    # Test double action (action 0 = double 1's)
    stochastic_action = 0
    new_state: State = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    assert jnp.array_equal(new_state._dice, jnp.array([0, 0]))  # type: ignore
    assert not new_state.is_stochastic  # type: ignore
    
    # Test non-double action (action 10 = 1,6)
    # In simple doubles mode, non-double actions should be ignored
    # and state should remain stochastic
    stochastic_action = 10
    original_dice = state._dice.copy()  # type: ignore
    new_state = env.stochastic_step(state, jnp.array(stochastic_action))  # type: ignore
    
    # Check that dice are unchanged for non-double action
    assert jnp.array_equal(new_state._dice, original_dice)  # type: ignore
    
    # State should still be stochastic since the action was ignored
    assert new_state.is_stochastic  # type: ignore

def test_stochastic_game_simulation():
    """Test simulating a game with predefined dice rolls."""
    # Create game environment
    env = Backgammon()
    
    # Initialize game
    state: State = env.init(jax.random.PRNGKey(42))  # type: ignore
    
    # Instead of random dice, we want to control the dice rolls
    # Let's simulate a short game with predefined dice
    
    # Test roll: 1,6 (action index 10)
    assert state.is_stochastic  # type: ignore
    state = env.stochastic_step(state, jnp.array(10))  # type: ignore
    assert not state.is_stochastic  # type: ignore
    assert jnp.array_equal(state._dice, jnp.array([0, 5]))  # type: ignore
    
    # Make a move using the first die
    legal_action = jnp.where(state.legal_action_mask)[0][0]
    state = env.step(state, legal_action, jax.random.PRNGKey(0))  # type: ignore
    
    # Make another move if the turn hasn't changed
    if state.is_stochastic:  # type: ignore
        # Turn has changed, and we need new dice
        # Let's roll doubles: 3,3 (action index 2)
        state = env.stochastic_step(state, jnp.array(2))  # type: ignore
        assert jnp.array_equal(state._dice, jnp.array([2, 2]))  # type: ignore
    else:
        # Make a move with the second die
        legal_action = jnp.where(state.legal_action_mask)[0][0]
        state = env.step(state, legal_action, jax.random.PRNGKey(1))  # type: ignore
        
        # Now the turn has likely changed, and we need new dice
        if state.is_stochastic:  # type: ignore
            # Let's roll 1,2 (action index 6)
            state = env.stochastic_step(state, jnp.array(6))  # type: ignore
            assert jnp.array_equal(state._dice, jnp.array([0, 1]))  # type: ignore
    
    # Verify we can continue making moves
    assert not state.terminated
    assert state.legal_action_mask.any()  # There should be legal actions available
