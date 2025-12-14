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

    @property
    def env_id(self) -> core.EnvId:
        return "pig"


class Pig(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)
        return _step(state, action, key)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "pig"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: PRNGKey) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int32(jax.random.bernoulli(subkey))
    legal_action_mask = jnp.zeros(6, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[0].set(True)
    return State(
        current_player=current_player,
        legal_action_mask=legal_action_mask,  # Must roll at start
        _scores=jnp.zeros(2, dtype=jnp.int32),
        _turn_total=jnp.int32(0)
    )  # type: ignore


def _step(state: State, action: Array, key) -> State:
    # action: 0 = Roll, 1 = Hold
    
    # Compute both outcomes
    state_roll = _roll(state, key)
    state_hold = _hold(state)
    
    # Select the correct outcome using jnp.where (mathematical multiplexing)
    is_roll = (action == 0)
    
    # We use tree_map to apply jnp.where to all fields in the State dataclass
    return jax.tree_util.tree_map(
        lambda r, h: jnp.where(is_roll, r, h),
        state_roll,
        state_hold
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
    roll = jax.random.randint(key, shape=(), minval=1, maxval=7)  # 1-6
    is_one = (roll == 1)
    
    # If roll is 1: turn total becomes 0.
    # If roll is 2-6: turn total becomes old + roll.
    # Logic: (old + roll) * (1 - is_one)
    new_turn_total = (state._turn_total + roll) * (1 - is_one)
    
    # If roll is 1: player flips.
    # If roll is 2-6: player stays.
    # Logic: (p + is_one) % 2
    new_player = (state.current_player + is_one) % 2
    
    return state.replace(
        current_player=new_player,
        _turn_total=new_turn_total,
        legal_action_mask=_get_legal_action_mask(new_turn_total)
    )


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
        legal_action_mask=_get_legal_action_mask(new_turn_total)
    )


def _observe(state: State, player_id: Array) -> Array:
    # [my_score, opp_score, turn_total] / 100.0
    scores = state._scores
    my_score = scores[player_id]
    opp_score = scores[1 - player_id]
    
    obs = jnp.array([my_score, opp_score, state._turn_total], dtype=jnp.float32)
    return obs / 100.0
