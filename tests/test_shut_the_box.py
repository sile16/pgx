import pgx
from pgx.shut_the_box import ShutTheBox, stochastic_action_to_str
import jax
import jax.numpy as jnp
import pytest

# --- Basic Tests ---

def test_init():
    env = ShutTheBox()
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    assert state.current_player == 0
    assert jnp.all(state._board == 1) # All open
    assert state._turn_sum == jnp.sum(state._dice + 1)
    assert not state.terminated
    assert state._is_stochastic # Initial state involves roll
    # At start, with all open, any roll 2-12 should be legal
    assert state.legal_action_mask.any()

def test_step_valid_move():
    env = ShutTheBox()
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # Force dice to be 3+6=9
    state = env.set_dice(state, jnp.array([2, 5], dtype=jnp.int32))
    assert state._turn_sum == 9
    
    # Action: Shut 9 (index 8, bitmask 256)
    # 2^8 = 256
    action = jnp.int32(256)
    
    assert state.legal_action_mask[action]
    
    state = env.step(state, action, key)
    
    # Check board update
    # Board index 8 should be 0 (shut)
    expected_board = jnp.ones(9, dtype=jnp.int32).at[8].set(0)
    assert jnp.all(state._board == expected_board)
    
    # Check reward: 9 points
    assert state.rewards[0] == 9.0
    
    # Check that new dice are rolled (or game over if unlucky)
    assert state._dice.shape == (2,)

def test_termination_no_moves():
    env = ShutTheBox()
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # Set board to almost shut: only 1 is open.
    # Dice roll 6 (3+3). Sum 6.
    # 1 is open, need 6. Impossible.
    
    board = jnp.zeros(9, dtype=jnp.int32).at[0].set(1) # Only 1 is open
    state = state.replace(_board=board)
    state = env.set_dice(state, jnp.array([2, 2], dtype=jnp.int32)) # 3+3=6
    
    assert state.terminated
    assert state.legal_action_mask.all() # Terminated state has all-true mask
    
    # Step on terminated state should return 0 reward
    state = env.step(state, jnp.int32(0), key)
    assert state.rewards[0] == 0.0

def test_perfect_game():
    env = ShutTheBox()
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # Manually shut everything
    # Board has only 9 open. Dice is 9.
    board = jnp.zeros(9, dtype=jnp.int32).at[8].set(1)
    state = state.replace(_board=board)
    state = env.set_dice(state, jnp.array([3, 4], dtype=jnp.int32)) # 4+5=9
    
    action = jnp.int32(256) # Shut 9
    assert state.legal_action_mask[action]
    
    state = env.step(state, action, key)
    
    assert jnp.all(state._board == 0)
    assert state.terminated
    assert state.rewards[0] == 9.0

def test_observation():
    env = ShutTheBox()
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # Set board: 1, 2 closed. 3-9 open.
    board = jnp.ones(9, dtype=jnp.int32).at[0].set(0).at[1].set(0)
    state = state.replace(_board=board)
    
    # Set dice: 1 and 1
    state = env.set_dice(state, jnp.array([0, 0], dtype=jnp.int32))
    
    obs = env.observe(state)
    
    # Check shapes
    assert obs.shape == (15,)
    
    # Check board part (0-8)
    assert obs[0] == 0
    assert obs[1] == 0
    assert obs[2] == 1
    
    # Check dice part (9-14)
    # Dice are 0 (value 1) and 0 (value 1).
    # Histogram: index 0 (value 1) should be 2. Others 0.
    assert obs[9] == 2
    assert obs[10] == 0

def test_api():
    env = ShutTheBox()
    assert env.id == "shut_the_box"
    assert env.num_players == 1
    assert env.num_actions == 512
    assert env.observation_shape == (15,)

# --- Stochastic Tests ---

def test_stochastic_step_transition():
    env = ShutTheBox()
    key = jax.random.PRNGKey(42)
    state = env.init(key)
    
    # Perform a normal step
    # Select a valid action
    action = jnp.argmax(state.legal_action_mask)
    state = env.step(state, action, key)
    
    if not state.terminated:
        # Should be stochastic again (new dice rolled)
        assert state._is_stochastic
    else:
        # If terminated, not stochastic (no new dice)
        assert not state._is_stochastic

def test_stochastic_step_override():
    env = ShutTheBox()
    key = jax.random.PRNGKey(42)
    state = env.init(key)
    
    # State has some random dice.
    # We want to force dice to be (6, 6) -> Sum 12.
    forced_state = env.stochastic_step(state, jnp.int32(35))
    
    assert jnp.all(forced_state._dice == jnp.array([5, 5], dtype=jnp.int32)) # 0-based
    assert forced_state._turn_sum == 12
    assert not forced_state._is_stochastic # Should be False after forcing
    
    # Verify legal action mask updated for sum 12
    # Mask should allow action corresponding to {3, 9} -> bitmask (1<<2)|(1<<8) = 4+256=260
    assert forced_state.legal_action_mask[260]

def test_stochastic_action_probs():
    env = ShutTheBox()
    probs = env.stochastic_action_probs
    assert probs.shape == (36,)
    assert jnp.allclose(jnp.sum(probs), 1.0)
    assert jnp.allclose(probs[0], 1/36)

def test_stochastic_action_str():
    assert stochastic_action_to_str(jnp.int32(0)) == "Rolled: 1-1"
    assert stochastic_action_to_str(jnp.int32(35)) == "Rolled: 6-6"
    assert stochastic_action_to_str(jnp.int32(1)) == "Rolled: 1-2"

# --- Extensive Tests ---

def test_api_compliance_manual():
    """Manual API compliance checks similar to what api_test does but safe for this env."""
    env = ShutTheBox()
    init = jax.jit(env.init)
    step = jax.jit(env.step)
    
    key = jax.random.PRNGKey(42)
    state = init(key)
    assert isinstance(state, pgx.State)
    assert state.legal_action_mask.shape == (512,)
    
    # Run a few steps
    for _ in range(10):
        if state.terminated:
            break
        # Pick random action
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        action = jax.random.categorical(key, logits)
        state = step(state, action, key)
        key, _ = jax.random.split(key)

def test_jit_compatibility():
    """Ensure init, step, and observe are JIT-able."""
    env = ShutTheBox()
    key = jax.random.PRNGKey(42)
    
    # JIT init
    jit_init = jax.jit(env.init)
    state = jit_init(key)
    assert isinstance(state, pgx.State)
    
    # JIT step
    jit_step = jax.jit(env.step)
    action = jnp.argmax(state.legal_action_mask)
    state = jit_step(state, action, key)
    assert isinstance(state, pgx.State)
    
    # JIT observe
    jit_observe = jax.jit(env.observe)
    obs = jit_observe(state)
    assert obs.shape == env.observation_shape

def test_vmap_compatibility():
    """Ensure environment works with vmap (batched execution)."""
    env = ShutTheBox()
    batch_size = 32
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    
    # Batch init
    v_init = jax.vmap(env.init)
    states = v_init(keys)
    assert states.current_player.shape == (batch_size,)
    
    # Batch step
    actions = jnp.argmax(states.legal_action_mask, axis=1)
    v_step = jax.vmap(env.step)
    states = v_step(states, actions, keys)
    
    assert states.rewards.shape == (batch_size, 1)

def test_legal_action_mask_correctness():
    """Verify legal action mask logic against explicit cases."""
    env = ShutTheBox()
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # Case 1: Sum 3, Board all open.
    state = env.set_dice(state, jnp.array([0, 1], dtype=jnp.int32)) # 1+2=3
    mask = state.legal_action_mask
    
    assert mask[3] # Binary 011 -> {1, 2}
    assert mask[4] # Binary 100 -> {3}
    assert not mask[5] # Binary 101 -> {1, 3} sum=4 != 3
    
    # Case 2: Sum 3, Tile 2 (value 3) is shut.
    board = jnp.ones(9, dtype=jnp.int32).at[2].set(0)
    state = state.replace(_board=board)
    state = env.set_dice(state, jnp.array([0, 1], dtype=jnp.int32)) # 1+2=3
    mask = state.legal_action_mask
    
    assert mask[3] # {1, 2} still valid
    assert not mask[4] # {3} now invalid because tile 3 is shut
    
def test_dice_distribution():
    """Check if dice rolls are roughly uniform."""
    env = ShutTheBox()
    key = jax.random.PRNGKey(999)
    batch_size = 1000
    
    states = jax.vmap(env.init)(jax.random.split(key, batch_size))
    actions = jnp.argmax(states.legal_action_mask, axis=1)
    next_states = jax.vmap(env.step)(states, actions, jax.random.split(key, batch_size))
    
    active_mask = ~next_states.terminated
    active_dice = next_states._dice[active_mask]
    
    if len(active_dice) > 0:
        mean_dice = jnp.mean(active_dice.astype(jnp.float32))
        assert 2.0 < mean_dice < 3.0
