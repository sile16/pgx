import jax
import jax.numpy as jnp
from pgx.play2048 import Play2048, _legal_action_mask, _slide_and_merge, State, stochastic_action_to_str

env = Play2048()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
slide_and_merge = jax.jit(_slide_and_merge)


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key=key)
    assert jnp.count_nonzero(state._board > 0) == 2
    assert jnp.isin(state._board[state._board > 0], jnp.int32([1, 2])).all()
    key = jax.random.PRNGKey(2)
    _, key = jax.random.split(key)  # for test compatibility
    state = init(key=key)
    assert state.legal_action_mask.shape == (4,)
    assert state.legal_action_mask.any()


def test_slide_and_merge():
    line = jnp.int32([0, 2, 0, 2])
    assert (slide_and_merge(line)[0] == jnp.int32([3, 0, 0, 0])).all()

    line = jnp.int32([0, 2, 0, 1])
    assert (slide_and_merge(line)[0] == jnp.int32([2, 1, 0, 0])).all()

    line = jnp.int32([2, 2, 2, 2])
    assert (slide_and_merge(line)[0] == jnp.int32([3, 3, 0, 0])).all()

    line = jnp.int32([2, 0, 0, 2])
    assert (slide_and_merge(line)[0] == jnp.int32([3, 0, 0, 0])).all()

    line = jnp.int32([1, 4, 4, 5])
    assert (slide_and_merge(line)[0] == jnp.int32([1, 5, 5, 0])).all()

    board = jnp.int32([0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
    board_2d = board.reshape((4, 4))
    board_2d = jax.vmap(_slide_and_merge)(board_2d)[0]
    board_1d = board_2d.ravel()
    assert (
        board_1d == jnp.int32([3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    ).all()


def test_step():
    board = jnp.zeros(16, dtype=jnp.int32).at[0].set(1)  # a single "2" tile at (0,0)
    state = State(_board=board, legal_action_mask=jnp.ones(4, dtype=jnp.bool_))

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    state1 = step(state, 3, key1)  # down
    state2 = step(state, 3, key2)  # down

    assert int(state1._board[12]) == 1  # tile moved to (3,0)
    assert int(state2._board[12]) == 1  # tile moved to (3,0)
    assert jnp.count_nonzero(state1._board > 0) == 2
    assert jnp.count_nonzero(state2._board > 0) == 2
    assert not (state1._board == state2._board).all()


def test_legal_action():
    board = jnp.int32([0, 1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 0])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(0))
    mask = _legal_action_mask(state._board.reshape((4, 4)))
    terminated = ~mask.any()
    mask = jax.lax.select(terminated, jnp.ones_like(mask), mask)
    assert (state.legal_action_mask == mask).all()
    assert state.terminated == terminated

    board = jnp.int32([2, 2, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(3))
    mask = _legal_action_mask(state._board.reshape((4, 4)))
    terminated = ~mask.any()
    mask = jax.lax.select(terminated, jnp.ones_like(mask), mask)
    assert (state.legal_action_mask == mask).all()
    assert state.terminated == terminated


def test_terminated():
    board = jnp.int32([1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 4, 5, 6])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(0))
    """
    [[ 2  4  8 16]
     [ 4  8 16 32]
     [ 8 16 32 64]
     [16 32 64  2]]
    """
    assert state.terminated


def test_observe():
    board = jnp.zeros(16, dtype=jnp.int32)
    board = board.at[2].set(1)  # (0,2) = 2
    board = board.at[3].set(2)  # (0,3) = 4
    state = State(_board=board)
    obs = observe(state, 0)
    assert obs.shape == (4, 4, 31)

    assert obs[0, 2, 1]
    assert obs[0, 3, 2]


def test_api():
    import pgx
    env = pgx.make("2048")
    pgx.api_test(env, 3, use_key=True)


def test_stochastic_flag_and_override():
    env = Play2048()
    key = jax.random.PRNGKey(0)
    _, key = jax.random.split(key)  # due to API update
    state = env.init(key)

    key, subkey = jax.random.split(key)
    state_after = env.step(state, jnp.int32(3), subkey)  # down
    assert state_after._is_stochastic

    base = state_after._stochastic_board
    diff = base != state_after._board
    assert int(diff.sum()) == 1
    spawn_pos = int(jnp.argmax(diff))
    assert int(base[spawn_pos]) == 0
    assert int(state_after._board[spawn_pos]) in (1, 2)

    original_num = int(state_after._board[spawn_pos])
    forced_value_idx = 1 if original_num == 1 else 0  # flip 2<->4
    forced_action = jnp.int32(2 * spawn_pos + forced_value_idx)
    forced_state = env.stochastic_step(state_after, forced_action)

    assert not forced_state._is_stochastic
    assert (forced_state.rewards == state_after.rewards).all()

    expected_num = jnp.int32(forced_value_idx + 1)
    expected_board = base.at[spawn_pos].set(expected_num)
    assert (forced_state._board == expected_board).all()

    expected_mask = _legal_action_mask(expected_board.reshape((4, 4)))
    expected_terminated = ~expected_mask.any()
    expected_mask = jax.lax.select(expected_terminated, jnp.ones_like(expected_mask), expected_mask)
    assert (forced_state.legal_action_mask == expected_mask).all()
    assert forced_state.terminated == expected_terminated


def test_stochastic_override_invalid_action_is_noop():
    env = Play2048()
    key = jax.random.PRNGKey(1)
    _, key = jax.random.split(key)  # due to API update
    state = env.init(key)

    key, subkey = jax.random.split(key)
    state_after = env.step(state, jnp.int32(0), subkey)  # left
    base = state_after._stochastic_board
    non_empty_pos = int(jnp.where(base != 0, size=1, fill_value=0)[0][0])

    invalid_action = jnp.int32(2 * non_empty_pos)  # try to place "2" on occupied cell
    state_invalid = env.stochastic_step(state_after, invalid_action)
    assert state_invalid._is_stochastic  # still overrideable
    assert (state_invalid._board == state_after._board).all()


def test_stochastic_step_vmap():
    env = Play2048()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    states = jax.vmap(env.init)(keys)

    keys2 = jax.random.split(jax.random.PRNGKey(1), 2)
    actions = jnp.int32([0, 3])
    states2 = jax.vmap(env.step)(states, actions, keys2)
    assert states2._is_stochastic.all()

    base = states2._stochastic_board
    diff = base != states2._board
    spawn_pos = jnp.argmax(diff, axis=1).astype(jnp.int32)
    spawn_num = jnp.take_along_axis(states2._board, spawn_pos[:, None], axis=1).squeeze(1)
    forced_value_idx = jnp.where(spawn_num == 1, 1, 0).astype(jnp.int32)
    forced_actions = (2 * spawn_pos + forced_value_idx).astype(jnp.int32)

    states3 = jax.vmap(env.stochastic_step)(states2, forced_actions)
    assert (~states3._is_stochastic).all()


def test_stochastic_action_to_str():
    assert stochastic_action_to_str(jnp.int32(0)) == "Spawned: 2 at (0,0)"
    assert stochastic_action_to_str(jnp.int32(1)) == "Spawned: 4 at (0,0)"
    assert stochastic_action_to_str(jnp.int32(31)) == "Spawned: 4 at (3,3)"
