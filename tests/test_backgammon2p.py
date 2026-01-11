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


def test_reverse_order_applied_when_required():
    """If only the reverse die order is legal, the env should apply that order."""
    env = Backgammon2P()
    # One checker on the bar and one on point 1; dice = (1, 2) sorted.
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[24].set(1)  # checker on bar
    board = board.at[0].set(1)   # checker on point 1
    state = env.init(jax.random.PRNGKey(0))
    state = state.replace(_board=board)
    state = env.set_dice(state, jnp.array([0, 1], dtype=jnp.int32))

    # Action encodes (src for die1=point1, src for die2=bar); only reverse order is legal.
    action = jnp.int32(2 * 26 + 1)
    assert bool(state.legal_action_mask[action])

    next_state = env.step_deterministic(state, action)
    # After applying reverse order, both checkers should land on point 2 (mirrored to index 22 after turn flip)
    # and bar should be empty.
    assert int(next_state._board[24]) == 0  # current player's bar empty
    assert int(next_state._board[25]) == 0  # opponent bar empty
    assert int(next_state._board[22]) == -2  # two opponent checkers at mirrored point 2


def test_random_short_games_terminate():
    """A small batch of random short-game episodes should finish within a modest step cap."""
    import jax

    env = Backgammon2P(short_game=True)
    batch_size = 4
    max_steps = 1000

    init_fn = jax.jit(jax.vmap(env.init))
    det_step = env.step_deterministic
    stoch_step = env.stochastic_step

    @jax.jit
    def run_batch(key):
        keys = jax.random.split(key, batch_size + 1)
        states = init_fn(keys[1:])

        steps = jnp.zeros(batch_size, dtype=jnp.int32)
        step_count = jnp.int32(0)
        key = keys[0]

        def cond(carry):
            states, key, steps, step_count = carry
            return (step_count < max_steps) & jnp.any(~states.terminated)

        def body(carry):
            states, key, steps, step_count = carry
            key, action_key = jax.random.split(key, 2)
            action_keys = jax.random.split(action_key, batch_size)

            def choose_action(state, k):
                return jax.lax.cond(
                    state._is_stochastic,
                    lambda: jax.random.randint(k, (), 0, env.num_stochastic_actions),
                    lambda: jax.random.categorical(k, logits=jnp.where(state.legal_action_mask, 0.0, -1e9)),
                )

            actions = jax.vmap(choose_action)(states, action_keys)

            def step_one(state, action):
                return jax.lax.cond(
                    state._is_stochastic,
                    lambda: stoch_step(state, action),
                    lambda: det_step(state, action),
                )

            next_states = jax.vmap(step_one)(states, actions)
            running = ~states.terminated
            steps = steps + running.astype(jnp.int32)
            return next_states, key, steps, step_count + 1

        final_states, _, steps_taken, _ = jax.lax.while_loop(cond, body, (states, key, steps, step_count))
        return final_states.terminated, steps_taken

    terminated, steps_taken = run_batch(jax.random.PRNGKey(0))
    assert bool(jnp.all(terminated)), f"Batch did not terminate; steps={steps_taken}"


def test_mask_zero_does_not_stall(monkeypatch):
    """If legality computation returns an all-zero mask, remaining_actions should drop to 0."""
    import jax

    env = Backgammon2P()

    def fake_legal_mask(board, dice):
        return jnp.zeros(26 * 26, dtype=jnp.bool_)

    monkeypatch.setattr("pgx.backgammon2p._legal_action_mask", fake_legal_mask)

    state = env.init(jax.random.PRNGKey(1))
    # Force deterministic phase with remaining action
    state = state.replace(_is_stochastic=jnp.array(False), _remaining_actions=jnp.int32(1))
    action = jnp.int32(0)  # pass (mask is zeroed anyway)
    next_state = env.step_deterministic(state, action)

    assert int(next_state._remaining_actions) == 0
    assert next_state.legal_action_mask.shape == (26 * 26,)


def test_auto_skip_turn_when_no_moves():
    """States with only the pass action should auto-advance without requiring a pass action."""
    env = Backgammon2P()
    state = env.init(jax.random.PRNGKey(2))
    pass_only_mask = jnp.zeros_like(state.legal_action_mask).at[0].set(True)
    state = state.replace(
        legal_action_mask=pass_only_mask,
        _is_stochastic=jnp.array(False),
        _remaining_actions=jnp.int32(1),
    )

    action = jnp.int32(42)  # arbitrary non-pass action
    next_state = env.step(state, action, jax.random.PRNGKey(9))

    assert not next_state.terminated
    assert int(next_state.current_player) != int(state.current_player)
    assert int(next_state._step_count) >= int(state._step_count) + 1
    assert next_state.legal_action_mask.any()


def test_step_stochastic_random_skips_pass_only(monkeypatch):
    """Stochastic helper should auto-advance until a playable mask exists."""
    env = Backgammon2P()
    pass_mask = jnp.zeros(26 * 26, dtype=jnp.bool_).at[0].set(True)
    playable_mask = jnp.zeros(26 * 26, dtype=jnp.bool_).at[5].set(True)

    def fake_chance_outcomes(state):
        return jnp.array([0], dtype=jnp.int32), jnp.array([1.0], dtype=jnp.float32)

    def fake_step_stochastic(state, outcome):
        # Always return a pass-only mask to force the auto-advance branch.
        return state.replace(_is_stochastic=False, legal_action_mask=pass_mask)

    auto_advance_calls = {"used": False}

    def fake_auto_advance(state, key):
        auto_advance_calls["used"] = True
        # Return a playable deterministic state so the loop exits.
        return state.replace(_is_stochastic=False, legal_action_mask=playable_mask)

    monkeypatch.setattr(env, "chance_outcomes", fake_chance_outcomes)
    monkeypatch.setattr(env, "_step_stochastic", fake_step_stochastic)
    monkeypatch.setattr(env, "_auto_advance_no_playable", fake_auto_advance)

    state = env.init(jax.random.PRNGKey(0))
    state = state.replace(_is_stochastic=jnp.array(True))

    next_state = env.step_stochastic_random(state, jax.random.PRNGKey(1))

    assert auto_advance_calls["used"]
    assert bool(next_state.legal_action_mask[5])
    assert not bool(next_state._is_stochastic)
