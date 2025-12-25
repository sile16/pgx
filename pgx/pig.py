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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(3, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.zeros(6, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Pig specific ---
    _scores: Array = jnp.zeros(2, dtype=jnp.int32)
    _turn_total: Array = jnp.int32(0)
    # Stochastic related
    _is_stochastic: Array = FALSE
    _last_roll: Array = jnp.int32(0)
    _prev_turn_total: Array = jnp.int32(0)

    @property
    def env_id(self) -> core.EnvId:
        return "pig"


class Pig(core.StochasticEnv):
    def __init__(self):
        super().__init__()
        # 1-6 (1/6 probability each)
        self.stochastic_action_probs = jnp.ones(6, dtype=jnp.float32) / 6.0

    def step(self, state: core.State, action: Array, key: Optional[Array] = None) -> core.State:
        assert key is not None, (
            "v2.0.0 changes the signature of step. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step(state, action)\n"
            "  * >= v2.0.0: step(state, action, key)\n\n"
            "See v2.0.0 release note for more details:\n\n"
            "  https://github.com/sotetsuk/pgx/releases/tag/v2.0.0"
        )
        return super().step(state, action, key)

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)
        return super()._step(state, action, key)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "pig"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 2

    def step_deterministic(self, state: State, action: Array) -> State:
        return _step_deterministic(state, action)

    def step_stochastic(self, state: State, action: Array) -> State:
        return _step_stochastic(state, action)

    def chance_outcomes(self, state: State) -> Tuple[Array, Array]:
        return jnp.arange(6, dtype=jnp.int32), self.stochastic_action_probs

    def stochastic_step(self, state: State, action: Array) -> State:
        # Backward compatibility
        return self.step_stochastic(state, action)


def _init(rng: PRNGKey) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int32(jax.random.bernoulli(subkey))
    legal_action_mask = jnp.zeros(6, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[0].set(True)
    return State(
        current_player=current_player,
        legal_action_mask=legal_action_mask,  # Must roll at start
        _scores=jnp.zeros(2, dtype=jnp.int32),
        _turn_total=jnp.int32(0),
        _is_stochastic=FALSE,
        _last_roll=jnp.int32(0),
        _prev_turn_total=jnp.int32(0),
    )  # type: ignore


def _step_deterministic(state: State, action: Array) -> State:
    # action: 0 = Roll, 1 = Hold
    
    # 1. Roll Chosen -> Transition to Chance Node
    state_roll = state.replace(
        _is_stochastic=TRUE,
        legal_action_mask=jnp.zeros(6, dtype=jnp.bool_),
        # turn_total remains unchanged (will be updated in stochastic step)
    )

    # 2. Hold Chosen -> Deterministic Update
    state_hold = _hold(state)
    
    return jax.lax.cond(
        action == 0,
        lambda: state_roll,
        lambda: state_hold
    )


def _step_stochastic(state: State, action: Array) -> State:
    # action: 0..5 (roll 1..6)
    roll = action + 1
    is_one = (roll == 1)
    
    new_turn_total = (state._turn_total + roll) * (1 - is_one)
    next_player = (state.current_player + is_one) % 2
    
    mask = _get_legal_action_mask(new_turn_total)
    
    return state.replace(
        current_player=next_player,
        _turn_total=new_turn_total,
        legal_action_mask=mask,
        _is_stochastic=FALSE,
        _last_roll=roll,
        _prev_turn_total=state._turn_total 
    )


def _decision_step(state: State, action: Array, key) -> State:
    # action: 0 = Roll, 1 = Hold
    return jax.lax.cond(
        action == 0,
        lambda: _prepare_roll(state),
        lambda: _hold(state)
    )


def _prepare_roll(state: State) -> State:
    # Transition to stochastic
    # legal actions for chance: 0..5 (1..6)
    return state.replace(
        _prev_turn_total=state._turn_total,
        _is_stochastic=TRUE,
        legal_action_mask=jnp.ones(6, dtype=jnp.bool_)
    )


def _chance_step(state: State, action: Array) -> State:
    # action 0..5 -> roll 1..6
    roll = action + 1
    is_one = (roll == 1)
    
    # Use saved prev total (or current, as it hasn't changed)
    prev_turn_total = state._turn_total
    
    new_turn_total = (prev_turn_total + roll) * (1 - is_one)
    new_player = (state.current_player + is_one) % 2
    
    return state.replace(
        current_player=new_player,
        _turn_total=new_turn_total,
        legal_action_mask=_get_legal_action_mask(new_turn_total),
        _is_stochastic=FALSE,
        _last_roll=roll,
        _prev_turn_total=prev_turn_total
    )


def _get_legal_action_mask(turn_total: Array) -> Array:
    # Action 0 (Roll) is always legal (True)
    # Action 1 (Hold) is legal if turn_total > 0
    # Logic: min(turn_total, 1) -> 0 if 0, 1 if > 0 (for positive integers)
    can_hold = jnp.minimum(turn_total, 1).astype(jnp.bool_)
    mask = jnp.zeros(6, dtype=jnp.bool_)
    mask = mask.at[0].set(True)
    mask = mask.at[1].set(can_hold)
    return mask


def _roll(state: State, key: PRNGKey) -> State:
    # Kept for potential internal usage, implemented via new primitives
    state = _step_deterministic(state, jnp.int32(0))
    roll = jax.random.randint(key, shape=(), minval=0, maxval=6)
    return _step_stochastic(state, roll)


def _hold(state: State) -> State:
    current_scores = state._scores
    player = state.current_player
    new_score = current_scores[player] + state._turn_total
    new_scores = current_scores.at[player].set(new_score)
    
    won = new_score >= 100
    
    # Rewards
    reward_val = won.astype(jnp.float32)
    rewards = jnp.zeros(2, dtype=jnp.float32)
    rewards = rewards.at[player].set(reward_val)
    rewards = rewards.at[1 - player].set(-reward_val)
    
    # Switch player if not won
    should_switch = ~won
    new_player = (player + should_switch) % 2
    
    # Reset turn total
    new_turn_total = jnp.int32(0)
    
    return state.replace(
        current_player=new_player,
        _scores=new_scores,
        _turn_total=new_turn_total,
        terminated=won,
        rewards=rewards,
        legal_action_mask=_get_legal_action_mask(new_turn_total),
        _is_stochastic=FALSE,
        _last_roll=jnp.int32(0),
        _prev_turn_total=jnp.int32(0)
    )

def _observe(state: State, player_id: Array) -> Array:
    # [my_score, opp_score, turn_total, is_stochastic]
    scores = state._scores
    my_score = scores[player_id]
    opp_score = scores[1 - player_id]
    
    obs = jnp.array([
        my_score / 100.0, 
        opp_score / 100.0, 
        state._turn_total / 100.0, 
        state._is_stochastic.astype(jnp.float32)
    ], dtype=jnp.float32)
    return obs

def stochastic_action_to_str(action: Array) -> str:
    """
    Convert a stochastic action (dice selection) to a human-readable string.
    action: 0..5 represents rolling 1..6
    """
    return f"Rolled: {int(action) + 1}"
