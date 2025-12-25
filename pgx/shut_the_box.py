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
    observation: Array = jnp.zeros(16, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.zeros(512, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Shut the Box specific ---
    # 9 tiles, 1=Open/Up, 0=Shut/Down
    _board: Array = jnp.ones(9, dtype=jnp.int32)
    _dice: Array = jnp.zeros(2, dtype=jnp.int32)  # 0-5 representing 1-6
    _turn_sum: Array = jnp.int32(0) # Current dice sum to match
    _is_stochastic: Array = FALSE

    @property
    def env_id(self) -> core.EnvId:
        return "shut_the_box"  # type: ignore


class ShutTheBox(core.StochasticEnv):
    def __init__(self):
        super().__init__()
        self.stochastic_action_probs = jnp.ones(36, dtype=jnp.float32) / 36.0

    def step(self, state: core.State, action: Array, key: Optional[Array] = None) -> core.State:
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
        return "shut_the_box"  # type: ignore

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return 512

    def step_deterministic(self, state: State, action: Array) -> State:
        return _step_deterministic(state, action)

    def step_stochastic(self, state: State, action: Array) -> State:
        # Calculate dice from action index
        d1 = action // 6
        d2 = action % 6
        dice = jnp.stack([d1, d2])
        return self.set_dice(state, dice)

    def chance_outcomes(self, state: State) -> Tuple[Array, Array]:
        outcomes = jnp.arange(36, dtype=jnp.int32)
        return outcomes, self.stochastic_action_probs

    def set_dice(self, state: State, dice: Array) -> State:
        """Use for testing or setting dice explicitly."""
        turn_sum = jnp.sum(dice + 1)
        legal_mask = _get_legal_action_mask(state._board, turn_sum)
        terminated = ~legal_mask.any()
        # If terminated, set mask to all True to comply with core.py expectations
        legal_mask = jax.lax.select(terminated, jnp.ones_like(legal_mask), legal_mask)
        return state.replace(
            _dice=dice,
            _turn_sum=turn_sum,
            legal_action_mask=legal_mask,
            terminated=terminated,
            _is_stochastic=FALSE  # Dice set, no longer stochastic
        )
    
    def stochastic_step(self, state: State, action: Array) -> State:
        return self.step_stochastic(state, action)


def _init(rng: PRNGKey) -> State:
    # Init returns an afterstate (chance node) where board is full but dice aren't rolled yet.
    # User Spec: "Initial state is stochastic, as it requires a dice roll"
    
    board = jnp.ones(9, dtype=jnp.int32)
    
    state = State(
        current_player=jnp.int32(0),
        _board=board,
        _dice=jnp.zeros(2, dtype=jnp.int32),
        _turn_sum=jnp.int32(0),
        legal_action_mask=jnp.zeros(512, dtype=jnp.bool_),
        terminated=FALSE,
        rewards=jnp.float32([0.0]),
        _is_stochastic=TRUE
    ) # type: ignore
    return state


def _step_deterministic(state: State, action: Array) -> State:
    # action is a bitmask of tiles to shut (0..511)
    tiles_to_shut = _action_to_tiles(action)
    
    tile_values = jnp.arange(1, 10, dtype=jnp.int32)
    step_reward = jnp.sum(tiles_to_shut * tile_values)
    
    new_board = state._board - tiles_to_shut
    
    # Check Win (All Shut)
    all_shut = jnp.all(new_board == 0)
    
    # After move, it's a chance node (dice roll) unless game ended.
    return state.replace(
        _board=new_board,
        rewards=jnp.array([step_reward], dtype=jnp.float32),
        terminated=all_shut,
        legal_action_mask=jnp.zeros(512, dtype=jnp.bool_),
        _is_stochastic=~all_shut
    ) # type: ignore


# --- Helpers ---

def _action_to_tiles(action: Array) -> Array:
    """Converts integer action (0-511) to binary array of size 9."""
    bits = (action[:, None] & (1 << jnp.arange(9))) > 0 if action.ndim > 0 else (action & (1 << jnp.arange(9))) > 0
    return bits.astype(jnp.int32)

def stochastic_action_to_str(action: Array) -> str:
    """
    Convert a stochastic action (dice roll index) to a human-readable string.
    action: 0..35
    """
    d1 = int(action // 6) + 1
    d2 = int(action % 6) + 1
    return f"Rolled: {d1}-{d2}"

_ALL_ACTIONS = jnp.arange(512, dtype=jnp.int32)
_ACTION_MASKS = ((_ALL_ACTIONS[:, None] & (1 << jnp.arange(9))) > 0).astype(jnp.int32)
_ACTION_SUMS = jnp.sum(_ACTION_MASKS * jnp.arange(1, 10, dtype=jnp.int32), axis=1)

def _get_legal_action_mask(board: Array, target_sum: Array) -> Array:
    is_available = jnp.all((_ACTION_MASKS & board) == _ACTION_MASKS, axis=1)
    is_sum_match = (_ACTION_SUMS == target_sum)
    return is_available & is_sum_match


def _observe(state: State, player_id: Array) -> Array:
    """
    Observation:
    [0-8]: Board (0=Shut, 1=Open)
    [9-14]: Dice Histogram (counts of 1..6)
    [15]: IS_CHANCE flag
    """
    board_obs = state._board.astype(jnp.float32)
    
    # Mask dice if stochastic
    dice_vals = jnp.arange(6)
    dice_hist = jax.lax.cond(
        state._is_stochastic,
        lambda: jnp.zeros(6, dtype=jnp.float32),
        lambda: jnp.sum(state._dice[:, None] == dice_vals[None, :], axis=0).astype(jnp.float32)
    )
    
    is_chance = state._is_stochastic.astype(jnp.float32)
    
    return jnp.concatenate([board_obs, dice_hist, jnp.array([is_chance])])