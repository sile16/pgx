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
    # After step, it's a decision node for NEXT turn
    mask = _legal_action_mask(state._board.reshape((4, 4)))
    assert (state.legal_action_mask == mask).all()

    board = jnp.int32([2, 2, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
    state = State(_board=board)
    state = step(state, 0, jax.random.PRNGKey(3))
    mask = _legal_action_mask(state._board.reshape((4, 4)))
    assert (state.legal_action_mask == mask).all()


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
    assert obs.shape == (4, 4, 32)

    assert obs[0, 2, 1]
    assert obs[0, 3, 2]
    assert not obs[0, 0, 31] # Not stochastic


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
    # Use split API to test flag
    afterstate = env.step_deterministic(state, jnp.int32(3))  # down
    assert afterstate._is_stochastic
    assert env.observe(afterstate)[0, 0, 31] # flag in obs

    base = afterstate._stochastic_board
    
    # Force a specific spawn via step_stochastic
    # Find an empty spot
    spawn_pos = int(jnp.where(base == 0, size=1, fill_value=0)[0][0])
    forced_action = jnp.int32(2 * spawn_pos + 1) # place a '4'
    forced_state = env.step_stochastic(afterstate, forced_action)

    assert not forced_state._is_stochastic
    assert int(forced_state._board[spawn_pos]) == 2 # set_num = value_idx + 1 = 1 + 1 = 2 (which is 2^2=4)
    # Wait, in 2048 code: set_num = (value_idx + 1). 1 -> 2, 2 -> 4. 
    # So value_idx=1 -> set_num=2 -> 2^2 = 4. Correct.

    assert (forced_state._board[spawn_pos] == 2)


def test_stochastic_step_vmap():
    env = Play2048()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    states = jax.vmap(env.init)(keys)

    actions = jnp.int32([0, 3])
    afterstates = jax.vmap(env.step_deterministic)(states, actions)
    assert afterstates._is_stochastic.all()

    # Force spawns
    forced_actions = jnp.int32([0, 2]) # pos 0 val 2, pos 1 val 2
    states_final = jax.vmap(env.step_stochastic)(afterstates, forced_actions)
    assert (~states_final._is_stochastic).all()