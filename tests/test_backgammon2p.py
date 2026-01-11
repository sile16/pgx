import jax
import jax.numpy as jnp

from pgx.backgammon2p import Backgammon2P


def _make_state_with_board(env, board, dice):
    state = env.init(jax.random.PRNGKey(0))
    state = state.replace(_board=board)
    return env.set_dice(state, dice)


def test_action_mask_shape():
    env = Backgammon2P()
    state = env.init(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (26 * 26,)


def test_single_move_prefers_higher_die():
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(1)
    board = board.at[1].set(-2)
    board = board.at[7].set(-2)
    dice = jnp.array([0, 5], dtype=jnp.int32)
    state = _make_state_with_board(env, board, dice)

    src_pass = 0
    src_point1 = 2
    action_pass_then_high = src_pass * 26 + src_point1
    action_low_then_pass = src_point1 * 26 + src_pass

    mask = state.legal_action_mask
    assert mask[action_pass_then_high]
    assert not mask[action_low_then_pass]
    assert mask.sum() == 1


def test_order_independent_sequence():
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[3].set(1)
    dice = jnp.array([1, 3], dtype=jnp.int32)
    state = _make_state_with_board(env, board, dice)

    src_point8 = 9
    src_point4 = 5
    action = src_point8 * 26 + src_point4
    assert state.legal_action_mask[action]


def test_doubles_use_two_actions():
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[0].set(1)
    board = board.at[3].set(1)
    dice = jnp.array([2, 2], dtype=jnp.int32)
    state = _make_state_with_board(env, board, dice)

    src_point1 = 2
    src_point4 = 5
    action = src_point1 * 26 + src_point4
    next_state = env.step(state, jnp.int32(action), jax.random.PRNGKey(1))
    assert int(next_state._remaining_actions) == 1
    assert not bool(next_state._is_stochastic)
