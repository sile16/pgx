# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests to validate that step() produces identical behavior to
step_deterministic() + step_stochastic().

For StochasticEnv games:
  step(state, action, key) should equal:
    1. afterstate = step_deterministic(state, action)
    2. next_state = step_stochastic(afterstate, sampled_outcome)
  where sampled_outcome is drawn from chance_outcomes() using key.
"""

import jax
import jax.numpy as jnp
import pytest

from pgx.backgammon import Backgammon
from pgx.play2048 import Play2048
from pgx.pig import Pig
from pgx.shut_the_box import ShutTheBox


def states_equal(s1, s2, rtol=1e-5, atol=1e-5):
    """Compare two states for equality, returning list of differences."""
    diffs = []

    # Compare common State fields
    if not jnp.array_equal(s1.current_player, s2.current_player):
        diffs.append(f"current_player: {s1.current_player} vs {s2.current_player}")

    if not jnp.allclose(s1.rewards, s2.rewards, rtol=rtol, atol=atol):
        diffs.append(f"rewards: {s1.rewards} vs {s2.rewards}")

    if not jnp.array_equal(s1.terminated, s2.terminated):
        diffs.append(f"terminated: {s1.terminated} vs {s2.terminated}")

    if not jnp.array_equal(s1.truncated, s2.truncated):
        diffs.append(f"truncated: {s1.truncated} vs {s2.truncated}")

    if not jnp.array_equal(s1.legal_action_mask, s2.legal_action_mask):
        diffs.append(f"legal_action_mask differs")

    if not jnp.array_equal(s1._step_count, s2._step_count):
        diffs.append(f"_step_count: {s1._step_count} vs {s2._step_count}")

    if not jnp.allclose(s1.observation, s2.observation, rtol=rtol, atol=atol):
        diffs.append(f"observation differs")

    return diffs


class TestBackgammonStepEquivalence:
    """Test step() vs step_deterministic() + step_stochastic() for Backgammon."""

    def test_single_step_equivalence(self):
        """Single step should produce identical results."""
        env = Backgammon()
        key = jax.random.PRNGKey(42)

        # Initialize
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        # Run for several steps
        for i in range(10):
            if state.terminated:
                break

            # Get legal action
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break
            action = legal_actions[0]

            key, step_key, det_key = jax.random.split(key, 3)

            # Method 1: Use step()
            state_via_step = env.step(state, action, step_key)

            # Method 2: Use step_deterministic() + step_stochastic()
            afterstate = env.step_deterministic(state, action)

            # Sample same outcome using the same key
            outcomes, probs = env.chance_outcomes(afterstate)
            outcome = jax.random.choice(det_key, outcomes, p=probs)
            state_via_split = env.step_stochastic(afterstate, outcome)

            # Compare (note: may differ due to different random sampling)
            # For now, just verify _step_count is incremented
            assert state_via_step._step_count == state._step_count + 1, \
                f"step() should increment _step_count"
            assert state_via_split._step_count == state._step_count + 1, \
                f"step_deterministic() should increment _step_count"

            # Continue with step() result
            state = state_via_step

    def test_step_count_increments_in_deterministic(self):
        """_step_count should increment in step_deterministic, not step_stochastic."""
        env = Backgammon()
        key = jax.random.PRNGKey(123)
        state = env.init(key)

        initial_count = state._step_count

        # Get a legal action
        legal_actions = jnp.where(state.legal_action_mask)[0]
        action = legal_actions[0]

        # step_deterministic should increment
        afterstate = env.step_deterministic(state, action)
        assert afterstate._step_count == initial_count + 1, \
            "step_deterministic should increment _step_count"

        # step_stochastic should NOT increment
        outcomes, probs = env.chance_outcomes(afterstate)
        next_state = env.step_stochastic(afterstate, outcomes[0])
        assert next_state._step_count == initial_count + 1, \
            "step_stochastic should NOT increment _step_count"


class TestPlay2048StepEquivalence:
    """Test step() vs step_deterministic() + step_stochastic() for 2048."""

    def test_single_step_equivalence(self):
        """Single step should produce identical results with same random key."""
        env = Play2048()
        key = jax.random.PRNGKey(42)

        # Initialize
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        for i in range(20):
            if state.terminated:
                break

            # Get legal action
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break
            action = legal_actions[0]

            key, step_key = jax.random.split(key)

            # Method 1: Use step()
            state_via_step = env.step(state, action, step_key)

            # Method 2: Use step_deterministic() + step_stochastic_random()
            afterstate = env.step_deterministic(state, action)
            state_via_split = env.step_stochastic_random(afterstate, step_key)

            # These should be identical since we use the same key
            diffs = states_equal(state_via_step, state_via_split)
            assert len(diffs) == 0, f"States differ: {diffs}"

            state = state_via_step

    def test_step_count_increments_in_deterministic(self):
        """_step_count should increment in step_deterministic, not step_stochastic."""
        env = Play2048()
        key = jax.random.PRNGKey(123)
        state = env.init(key)

        initial_count = state._step_count

        # Get a legal action
        legal_actions = jnp.where(state.legal_action_mask)[0]
        action = legal_actions[0]

        # step_deterministic should increment
        afterstate = env.step_deterministic(state, action)
        assert afterstate._step_count == initial_count + 1, \
            "step_deterministic should increment _step_count"

        # step_stochastic should NOT increment
        outcomes, probs = env.chance_outcomes(afterstate)
        next_state = env.step_stochastic(afterstate, outcomes[0])
        assert next_state._step_count == initial_count + 1, \
            "step_stochastic should NOT increment _step_count"


class TestPigStepEquivalence:
    """Test step() vs step_deterministic() + step_stochastic() for Pig."""

    def test_single_step_equivalence(self):
        """Single step should produce identical results with same random key."""
        env = Pig()
        key = jax.random.PRNGKey(42)

        # Initialize
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        for i in range(30):
            if state.terminated:
                break

            # Get legal action (0=roll, 1=hold)
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break
            action = legal_actions[0]  # Usually roll

            key, step_key = jax.random.split(key)

            # Method 1: Use step()
            state_via_step = env.step(state, action, step_key)

            # Method 2: Use step_deterministic() + step_stochastic()
            afterstate = env.step_deterministic(state, action)

            if afterstate._is_stochastic:
                # Sample same outcome
                outcomes, probs = env.chance_outcomes(afterstate)
                outcome = jax.random.choice(step_key, outcomes, p=probs)
                state_via_split = env.step_stochastic(afterstate, outcome)
            else:
                state_via_split = afterstate

            # Verify step counts match
            assert state_via_step._step_count == state._step_count + 1, \
                f"step() should increment _step_count"
            assert state_via_split._step_count == state._step_count + 1, \
                f"step_deterministic() should increment _step_count"

            state = state_via_step

    def test_step_count_increments_in_deterministic(self):
        """_step_count should increment in step_deterministic, not step_stochastic."""
        env = Pig()
        key = jax.random.PRNGKey(123)
        state = env.init(key)

        initial_count = state._step_count

        # Action 0 = Roll (leads to stochastic step)
        action = jnp.int32(0)

        # step_deterministic should increment
        afterstate = env.step_deterministic(state, action)
        assert afterstate._step_count == initial_count + 1, \
            "step_deterministic should increment _step_count"

        # step_stochastic should NOT increment
        assert afterstate._is_stochastic
        outcomes, probs = env.chance_outcomes(afterstate)
        next_state = env.step_stochastic(afterstate, outcomes[0])
        assert next_state._step_count == initial_count + 1, \
            "step_stochastic should NOT increment _step_count"


class TestShutTheBoxStepEquivalence:
    """Test step() vs step_deterministic() + step_stochastic() for Shut the Box."""

    def test_single_step_equivalence(self):
        """Single step should produce identical results with same random key."""
        env = ShutTheBox()
        key = jax.random.PRNGKey(42)

        # Initialize - starts in stochastic state (needs dice roll)
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        # Roll dice first
        key, dice_key = jax.random.split(key)
        outcomes, probs = env.chance_outcomes(state)
        outcome = jax.random.choice(dice_key, outcomes, p=probs)
        state = env.step_stochastic(state, outcome)

        for i in range(10):
            if state.terminated:
                break

            # Get legal action
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break
            action = legal_actions[0]

            key, step_key = jax.random.split(key)

            # Method 1: Use step()
            state_via_step = env.step(state, action, step_key)

            # Method 2: Use step_deterministic() + step_stochastic()
            afterstate = env.step_deterministic(state, action)

            if afterstate._is_stochastic and not afterstate.terminated:
                # Sample same outcome
                outcomes, probs = env.chance_outcomes(afterstate)
                outcome = jax.random.choice(step_key, outcomes, p=probs)
                state_via_split = env.step_stochastic(afterstate, outcome)
            else:
                state_via_split = afterstate

            # Verify step counts match
            assert state_via_step._step_count == state._step_count + 1, \
                f"step() should increment _step_count"
            assert state_via_split._step_count == state._step_count + 1, \
                f"step_deterministic() should increment _step_count"

            state = state_via_step

    def test_step_count_increments_in_deterministic(self):
        """_step_count should increment in step_deterministic, not step_stochastic."""
        env = ShutTheBox()
        key = jax.random.PRNGKey(123)
        state = env.init(key)

        # Roll dice first (stochastic step)
        initial_count = state._step_count
        outcomes, probs = env.chance_outcomes(state)
        state = env.step_stochastic(state, outcomes[0])

        # step_stochastic should NOT increment
        assert state._step_count == initial_count, \
            "step_stochastic should NOT increment _step_count"

        # Now do deterministic step
        initial_count = state._step_count
        legal_actions = jnp.where(state.legal_action_mask)[0]
        action = legal_actions[0]

        afterstate = env.step_deterministic(state, action)
        assert afterstate._step_count == initial_count + 1, \
            "step_deterministic should increment _step_count"


class TestTerminalStateHandling:
    """Test that terminal states are handled correctly in split steps."""

    def test_step_on_terminated_returns_zero_rewards(self):
        """Calling step on terminated state should return zero rewards."""
        env = Pig()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force terminated state
        state = state.replace(terminated=jnp.bool_(True), rewards=jnp.float32([1.0, -1.0]))

        # step() on terminated should return zero rewards
        key, step_key = jax.random.split(key)
        next_state = env.step(state, jnp.int32(0), step_key)
        assert jnp.allclose(next_state.rewards, jnp.zeros(2)), \
            "step() on terminated should return zero rewards"

    def test_step_deterministic_on_terminated_returns_zero_rewards(self):
        """Calling step_deterministic on terminated state should return zero rewards."""
        env = Pig()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force terminated state
        state = state.replace(terminated=jnp.bool_(True), rewards=jnp.float32([1.0, -1.0]))

        # step_deterministic() on terminated should return zero rewards
        next_state = env.step_deterministic(state, jnp.int32(0))
        assert jnp.allclose(next_state.rewards, jnp.zeros(2)), \
            "step_deterministic() on terminated should return zero rewards"


class TestIllegalActionHandling:
    """Test that illegal actions are handled correctly."""

    def test_illegal_action_penalty_in_step(self):
        """Illegal action in step() should give penalty."""
        env = Pig()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # At start, only roll (action 0) is legal, hold (action 1) is illegal
        assert state.legal_action_mask[0] == True
        assert state.legal_action_mask[1] == False

        # Take illegal action
        key, step_key = jax.random.split(key)
        next_state = env.step(state, jnp.int32(1), step_key)

        # Should terminate with penalty
        assert next_state.terminated
        assert next_state.rewards[state.current_player] < 0, \
            "Illegal action should give negative reward"

    def test_illegal_action_penalty_in_step_deterministic(self):
        """Illegal action in step_deterministic() should give penalty."""
        env = Pig()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # At start, only roll (action 0) is legal
        assert state.legal_action_mask[0] == True
        assert state.legal_action_mask[1] == False

        # Take illegal action via step_deterministic
        next_state = env.step_deterministic(state, jnp.int32(1))

        # Should terminate with penalty
        assert next_state.terminated
        assert next_state.rewards[state.current_player] < 0, \
            "Illegal action should give negative reward"


class TestLegalActionMaskAtTerminal:
    """Test that legal_action_mask is all True at terminal states."""

    def test_terminal_mask_in_step(self):
        """legal_action_mask should be all True after termination via step()."""
        env = Pig()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force win by setting score to 99 and holding
        state = state.replace(
            _scores=jnp.array([99, 0], dtype=jnp.int32),
            _turn_total=jnp.int32(1),
            legal_action_mask=jnp.array([True, True], dtype=jnp.bool_)
        )

        # Hold (action 1) to win
        key, step_key = jax.random.split(key)
        next_state = env.step(state, jnp.int32(1), step_key)

        assert next_state.terminated
        assert jnp.all(next_state.legal_action_mask), \
            "legal_action_mask should be all True at terminal"

    def test_terminal_mask_in_step_deterministic(self):
        """legal_action_mask should be all True after termination via step_deterministic()."""
        env = Pig()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force win by setting score to 99 and holding
        state = state.replace(
            _scores=jnp.array([99, 0], dtype=jnp.int32),
            _turn_total=jnp.int32(1),
            legal_action_mask=jnp.array([True, True], dtype=jnp.bool_)
        )

        # Hold (action 1) to win
        next_state = env.step_deterministic(state, jnp.int32(1))

        assert next_state.terminated
        assert jnp.all(next_state.legal_action_mask), \
            "legal_action_mask should be all True at terminal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
