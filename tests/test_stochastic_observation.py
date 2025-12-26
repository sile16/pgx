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
Test that StochasticEnv.step_deterministic and step_stochastic
properly update state.observation.

This is a regression test for the bug where calling these methods directly
would leave state.observation stale (not reflecting the new state).
"""

import jax
import jax.numpy as jnp
import pytest

import pgx


class TestStochasticObservationUpdate:
    """Tests that step_deterministic and step_stochastic update observations."""

    def test_pig_step_deterministic_updates_observation(self):
        """Pig: step_deterministic should update observation."""
        env = pgx.make("pig")
        state = env.init(jax.random.PRNGKey(42))

        # Roll (action=0) to create a chance node
        state2 = env.step_deterministic(state, jnp.int32(0))

        expected_obs = env.observe(state2)
        assert jnp.allclose(state2.observation, expected_obs), \
            f"Pig step_deterministic: obs={state2.observation}, expected={expected_obs}"

    def test_pig_step_stochastic_updates_observation(self):
        """Pig: step_stochastic should update observation."""
        env = pgx.make("pig")
        state = env.init(jax.random.PRNGKey(42))

        # Roll to get to chance node
        state2 = env.step_deterministic(state, jnp.int32(0))

        # Force roll 6 (action=5)
        state3 = env.step_stochastic(state2, jnp.int32(5))

        expected_obs = env.observe(state3)
        assert jnp.allclose(state3.observation, expected_obs), \
            f"Pig step_stochastic: obs={state3.observation}, expected={expected_obs}"

    def test_2048_step_deterministic_updates_observation(self):
        """2048: step_deterministic should update observation."""
        env = pgx.make("2048")
        state = env.init(jax.random.PRNGKey(42))

        # Left swipe
        state2 = env.step_deterministic(state, jnp.int32(0))

        expected_obs = env.observe(state2)
        assert jnp.allclose(state2.observation, expected_obs), \
            f"2048 step_deterministic: obs mismatch"

    def test_2048_step_stochastic_updates_observation(self):
        """2048: step_stochastic should update observation."""
        env = pgx.make("2048")
        state = env.init(jax.random.PRNGKey(42))

        # Left swipe to create afterstate
        state2 = env.step_deterministic(state, jnp.int32(0))

        # Find empty position and spawn tile
        empty_idx = int(jnp.argmax(state2._stochastic_board == 0))
        state3 = env.step_stochastic(state2, jnp.int32(2 * empty_idx))

        expected_obs = env.observe(state3)
        assert jnp.allclose(state3.observation, expected_obs), \
            f"2048 step_stochastic: obs mismatch"

    def test_2048_step_stochastic_random_updates_observation(self):
        """2048: step_stochastic_random should update observation."""
        env = pgx.make("2048")
        state = env.init(jax.random.PRNGKey(42))

        # Left swipe to create afterstate
        state2 = env.step_deterministic(state, jnp.int32(0))

        # Random spawn
        state3 = env.step_stochastic_random(state2, jax.random.PRNGKey(123))

        expected_obs = env.observe(state3)
        assert jnp.allclose(state3.observation, expected_obs), \
            f"2048 step_stochastic_random: obs mismatch"

    def test_shut_the_box_step_stochastic_updates_observation(self):
        """Shut the Box: step_stochastic should update observation."""
        env = pgx.make("shut_the_box")
        state = env.init(jax.random.PRNGKey(42))

        # Initial state is stochastic (needs dice roll)
        assert state._is_stochastic

        # Roll dice (action=0 is 1-1)
        state2 = env.step_stochastic(state, jnp.int32(0))

        expected_obs = env.observe(state2)
        assert jnp.allclose(state2.observation, expected_obs), \
            f"Shut the Box step_stochastic: obs mismatch"

    def test_shut_the_box_set_dice_updates_observation(self):
        """Shut the Box: set_dice should update observation."""
        env = pgx.make("shut_the_box")
        state = env.init(jax.random.PRNGKey(42))

        # Set specific dice
        state2 = env.set_dice(state, jnp.array([2, 3], dtype=jnp.int32))

        expected_obs = env.observe(state2)
        assert jnp.allclose(state2.observation, expected_obs), \
            f"Shut the Box set_dice: obs mismatch"

    def test_backgammon_step_stochastic_updates_observation(self):
        """Backgammon: step_stochastic should update observation."""
        env = pgx.make("backgammon")
        state = env.init(jax.random.PRNGKey(42))

        # Initial state is stochastic (needs dice roll)
        assert state._is_stochastic

        # Roll dice (action=5 is 2-1)
        state2 = env.step_stochastic(state, jnp.int32(5))

        expected_obs = env.observe(state2)
        assert jnp.allclose(state2.observation, expected_obs), \
            f"Backgammon step_stochastic: obs mismatch"

    def test_backgammon_step_deterministic_updates_observation(self):
        """Backgammon: step_deterministic should update observation."""
        env = pgx.make("backgammon")
        state = env.init(jax.random.PRNGKey(42))

        # Roll dice first
        state2 = env.step_stochastic(state, jnp.int32(5))

        # Make a move
        action = jnp.argmax(state2.legal_action_mask)
        state3 = env.step_deterministic(state2, action)

        expected_obs = env.observe(state3)
        assert jnp.allclose(state3.observation, expected_obs), \
            f"Backgammon step_deterministic: obs mismatch"

    def test_backgammon_set_dice_updates_observation(self):
        """Backgammon: set_dice should update observation."""
        env = pgx.make("backgammon")
        state = env.init(jax.random.PRNGKey(42))

        # Set specific dice
        state2 = env.set_dice(state, jnp.array([0, 1], dtype=jnp.int32))

        expected_obs = env.observe(state2)
        assert jnp.allclose(state2.observation, expected_obs), \
            f"Backgammon set_dice: obs mismatch"

    def test_normal_step_still_works(self):
        """Verify that normal step() still works correctly for all stochastic games."""
        games = ["pig", "2048", "shut_the_box", "backgammon"]

        for game_id in games:
            env = pgx.make(game_id)
            key = jax.random.PRNGKey(42)
            state = env.init(key)

            # Take a few steps
            for i in range(5):
                if state.terminated:
                    break
                key, subkey = jax.random.split(key)
                action = jax.random.choice(
                    subkey,
                    jnp.arange(state.legal_action_mask.shape[0]),
                    p=state.legal_action_mask / state.legal_action_mask.sum()
                )
                key, subkey = jax.random.split(key)
                state = env.step(state, action, subkey)

                expected_obs = env.observe(state)
                assert jnp.allclose(state.observation, expected_obs), \
                    f"{game_id} step {i}: observation mismatch after step()"
