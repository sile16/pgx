import pgx.pig as pig
import pgx.core as core
import jax
import jax.numpy as jnp
import pytest


def test_init():
    env = pig.Pig()
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    # Check initial state properties
    assert state.current_player in [0, 1]
    assert jnp.all(state.rewards == jnp.array([0.0, 0.0]))
    assert not state.terminated
    assert jnp.all(state._scores == jnp.array([0, 0]))
    assert state._turn_total == 0
    # At the start, only roll (action 0) should be legal
    assert state.legal_action_mask[0] == True
    assert jnp.all(state.legal_action_mask[1:] == False)
    # Check stochastic fields
    assert state._is_stochastic == False
    assert state._last_roll == 0
    assert state._prev_turn_total == 0

def test_step_roll():
    env = pig.Pig()
    key = jax.random.PRNGKey(4) # This key will give roll=1 after one split
    state = env.init(key)

    # Force a specific current player and roll for deterministic test
    # (In a real test, you might use a specific key or mock jax.random)
    state = state.replace(current_player=jnp.int32(0))

    # Test rolling a 1 (losing turn)
    key, subkey = jax.random.split(key) # subkey will now produce roll=1
    state = env.step(state, jnp.int32(0), subkey) # Action 0: Roll

    assert state.current_player == 1 # Player should switch
    assert state._turn_total == 0    # Turn total should reset
    assert jnp.all(state._scores == jnp.array([0, 0]))
    assert not state.terminated
    assert state.legal_action_mask[0] == True # Can roll again
    assert state.legal_action_mask[1] == False # Cannot hold with 0 turn_total
    
    # Check stochastic fields
    assert state._is_stochastic == True
    assert state._last_roll == 1
    assert state._prev_turn_total == 0

def test_step_hold():
    env = pig.Pig()
    key = jax.random.PRNGKey(0) # Use a fresh key
    state = env.init(key)

    # Force a specific current player and turn total
    state = state.replace(current_player=jnp.int32(0), _turn_total=jnp.int32(10), legal_action_mask=jnp.array([True, True, False, False, False, False]))

    state = env.step(state, jnp.int32(1), key) # Action 1: Hold

    assert state.current_player == 1 # Player should switch
    assert state._turn_total == 0    # Turn total should reset
    assert jnp.all(state._scores == jnp.array([10, 0])) # Player 0 score should be 10
    assert not state.terminated
    assert state.legal_action_mask[0] == True # Can roll again
    assert state.legal_action_mask[1] == False # Cannot hold with 0 turn_total
    
    # Check stochastic fields (should be reset)
    assert state._is_stochastic == False
    assert state._last_roll == 0

def test_win_condition():
    env = pig.Pig()
    key = jax.random.PRNGKey(0) # Use a fresh key
    state = env.init(key)

    # Set score to near winning, then hold to win
    state = state.replace(current_player=jnp.int32(0), _scores=jnp.array([95, 0]), _turn_total=jnp.int32(10), legal_action_mask=jnp.array([True, True, False, False, False, False]))
    state = env.step(state, jnp.int32(1), key) # Action 1: Hold

    assert state.terminated
    assert jnp.all(state.rewards == jnp.array([1.0, -1.0])) # Player 0 wins

def test_step_on_terminated_state_returns_zero_reward():
    env = pig.Pig()
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    # Make the game terminate deterministically: score 99, hold 1 point.
    state = state.replace(
        current_player=jnp.int32(0),
        _scores=jnp.array([99, 0]),
        _turn_total=jnp.int32(1),
        legal_action_mask=jnp.array([True, True, False, False, False, False]),
    )
    key, subkey = jax.random.split(key)
    state = env.step(state, jnp.int32(1), subkey)  # Hold -> win
    assert state.terminated
    assert jnp.all(state.rewards == jnp.array([1.0, -1.0]))
    assert state.legal_action_mask.all()

    # Stepping a terminated state should have no effect and yields zero rewards.
    prev_scores = state._scores
    key, subkey = jax.random.split(key)
    state2 = env.step(state, jnp.int32(0), subkey)
    assert state2.terminated
    assert jnp.all(state2.rewards == jnp.array([0.0, 0.0]))
    assert state2.legal_action_mask.all()
    assert jnp.all(state2._scores == prev_scores)

def test_observe():
    env = pig.Pig()
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    state = state.replace(current_player=jnp.int32(0), _scores=jnp.array([50, 20]), _turn_total=jnp.int32(15))

    # Observe from player 0's perspective
    obs0 = env.observe(state.replace(current_player=jnp.int32(0)))
    assert jnp.allclose(obs0, jnp.array([50/100, 20/100, 15/100]))

    # Observe from player 1's perspective
    obs1 = env.observe(state.replace(current_player=jnp.int32(1)))
    assert jnp.allclose(obs1, jnp.array([20/100, 50/100, 15/100]))

def test_stochastic_step():
    env = pig.Pig()
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # 1. Start with player 0, total 10
    state = state.replace(current_player=jnp.int32(0), _turn_total=jnp.int32(10))
    
    # 2. Perform a normal step (Roll)
    # This will generate some random roll.
    key, subkey = jax.random.split(key)
    state = env.step(state, jnp.int32(0), subkey)
    
    # Check it became stochastic
    assert state._is_stochastic == True
    original_roll = state._last_roll
    
    # 3. Override with stochastic_step (Force roll=6)
    # Action 5 -> Roll 6
    state_force_6 = env.stochastic_step(state, jnp.int32(5))
    
    assert state_force_6._is_stochastic == False
    assert state_force_6._last_roll == 6
    assert state_force_6._turn_total == 10 + 6
    assert state_force_6.current_player == 0 # Player stays
    
    # 4. Override with stochastic_step (Force roll=1)
    # Action 0 -> Roll 1
    state_force_1 = env.stochastic_step(state, jnp.int32(0))
    
    assert state_force_1._is_stochastic == False
    assert state_force_1._last_roll == 1
    assert state_force_1._turn_total == 0
    assert state_force_1.current_player == 1 # Player switches
    
    # 5. Check if we can override a "Rolled 1" state
    # Create a state that resulted from rolling 1
    # Previous total was 10. Rolled 1 -> total 0, player 1.
    state_rolled_1 = state_force_1
    assert state_rolled_1._prev_turn_total == 10
    
    # Now try to override this "Rolled 1" state to "Rolled 6"
    state_override_1_to_6 = env.stochastic_step(state_rolled_1, jnp.int32(5))
    
    assert state_override_1_to_6._last_roll == 6
    assert state_override_1_to_6._turn_total == 10 + 6 # Should recover previous total
    assert state_override_1_to_6.current_player == 0 # Should revert player switch
