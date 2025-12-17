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

from typing import Optional

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
    observation: Array = jnp.zeros(15, dtype=jnp.float32)
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


class ShutTheBox(core.Env):
    def __init__(self):
        super().__init__()
        self.stochastic_action_probs = jnp.ones(36, dtype=jnp.float32) / 36.0

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
        return _step(state, action, key)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "shut_the_box"  # type: ignore

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return 512

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
        """
        Force a specific dice roll based on action index 0-35.
        0 -> (1, 1), 1 -> (1, 2), ..., 5 -> (1, 6)
        6 -> (2, 1), ...
        35 -> (6, 6)
        """
        # Calculate dice from action index
        d1 = action // 6
        d2 = action % 6
        dice = jnp.stack([d1, d2])
        return self.set_dice(state, dice)


def _init(rng: PRNGKey) -> State:
    rng, subkey = jax.random.split(rng)
    # Roll initial dice
    dice = jax.random.randint(subkey, shape=(2,), minval=0, maxval=6)
    turn_sum = jnp.sum(dice + 1)
    
    board = jnp.ones(9, dtype=jnp.int32)
    
    # Check legal moves
    legal_action_mask = _get_legal_action_mask(board, turn_sum)
    
    # If no legal moves at start (rare but possible if we had constraints, though with 9 open tiles and max sum 12, always possible),
    # actually max sum 12 is always achievable with {1..9} unless board is weird.
    # With all {1..9} open, any sum 2..12 is possible.
    
    terminated = ~legal_action_mask.any()
    
    state = State(
        current_player=jnp.int32(0),
        _board=board,
        _dice=dice,
        _turn_sum=turn_sum,
        legal_action_mask=legal_action_mask,
        terminated=terminated,
        rewards=jnp.float32([0.0]), # No reward on init
        _is_stochastic=TRUE # Initial state involved a roll
    ) # type: ignore
    return state


def _step(state: State, action: Array, key) -> State:
    # 1. Apply Action
    # action is a bitmask of tiles to shut (0..511)
    # We assume action is legal (checked by engine/user, or we use cond)
    # To be safe, we can multiply by legal mask or assume it's valid.
    # PGX usually assumes valid action if selected from mask.
    
    # Convert action to tile array (0/1)
    tiles_to_shut = _action_to_tiles(action)
    
    # Reward = sum of tiles shut
    # Tiles are 1-based indices. 
    # tiles_to_shut is binary array size 9.
    # Values: 1, 2, ..., 9
    tile_values = jnp.arange(1, 10, dtype=jnp.int32)
    step_reward = jnp.sum(tiles_to_shut * tile_values)
    
    # Update Board: 1 (Open) -> 0 (Shut)
    # new_board = old_board - tiles_to_shut (since tiles_to_shut are 1 where we want to shut)
    new_board = state._board - tiles_to_shut
    
    # Check Win (All Shut)
    all_shut = jnp.all(new_board == 0)
    
    # Prepare next state variables
    def _continue_game(b, k):
        # Roll new dice
        new_dice = jax.random.randint(k, shape=(2,), minval=0, maxval=6)
        new_sum = jnp.sum(new_dice + 1)
        # Check legal moves
        mask = _get_legal_action_mask(b, new_sum)
        # Check termination (no moves)
        term = ~mask.any()
        return new_dice, new_sum, mask, term, ~term

    def _win_game():
        # Zeros for dice/sum, mask all false
        return jnp.zeros(2, dtype=jnp.int32), jnp.int32(0), jnp.zeros(512, dtype=jnp.bool_), TRUE, FALSE

    # Logic branching
    # If already won (all_shut), game ends.
    # Else, roll and check if can continue.
    
    new_dice, new_sum, legal_mask, terminated, is_stochastic = jax.lax.cond(
        all_shut,
        _win_game,
        lambda: _continue_game(new_board, key)
    )
    
    return state.replace(
        _board=new_board,
        _dice=new_dice,
        _turn_sum=new_sum,
        legal_action_mask=legal_mask,
        terminated=terminated,
        rewards=jnp.array([step_reward], dtype=jnp.float32),
        _is_stochastic=is_stochastic
    ) # type: ignore


# --- Helpers ---

def _action_to_tiles(action: Array) -> Array:
    """Converts integer action (0-511) to binary array of size 9."""
    # Powers of 2: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Bitwise operations to extract bits
    bits = (action[:, None] & (1 << jnp.arange(9))) > 0 if action.ndim > 0 else (action & (1 << jnp.arange(9))) > 0
    return bits.astype(jnp.int32)

def _tiles_to_sum(tiles: Array) -> Array:
    """Calculates sum of selected tiles."""
    values = jnp.arange(1, 10, dtype=jnp.int32)
    return jnp.sum(tiles * values, axis=-1)

def stochastic_action_to_str(action: Array) -> str:
    """
    Convert a stochastic action (dice roll index) to a human-readable string.
    action: 0..35
    """
    d1 = int(action // 6) + 1
    d2 = int(action % 6) + 1
    return f"Rolled: {d1}-{d2}"

# Precompute lookup tables for efficiency
# _ACTION_MASKS: (512, 9) binary
# _ACTION_SUMS: (512,) int
_ALL_ACTIONS = jnp.arange(512, dtype=jnp.int32)
_ACTION_MASKS = ((_ALL_ACTIONS[:, None] & (1 << jnp.arange(9))) > 0).astype(jnp.int32)
_ACTION_SUMS = jnp.sum(_ACTION_MASKS * jnp.arange(1, 10, dtype=jnp.int32), axis=1)

def _get_legal_action_mask(board: Array, target_sum: Array) -> Array:
    """
    board: (9,) int (1=Open, 0=Shut)
    target_sum: int
    """
    # 1. Check subset availability
    # Action mask requires tiles. Those tiles must be available in board.
    # If action requires tile i (mask[i]=1), board[i] must be 1.
    # Logic: (mask & ~board) == 0  --> No bit set in mask is unset in board.
    # Or simply: (mask <= board) since 0<=0, 0<=1, 1<=1, but 1 > 0 (fail).
    # Since they are binary 0/1 integers:
    # We need mask[i] == 1 => board[i] == 1.
    # which is equivalent to: mask & board == mask
    
    # Broadcast board to (512, 9)
    # _ACTION_MASKS is (512, 9)
    # We check if (_ACTION_MASKS * board) == _ACTION_MASKS
    # Wait, simple bitwise check on integer representation is faster?
    # But board is array (9,).
    # Let's stick to array ops.
    
    # Compatibility check
    # mask (1s are required)
    # board (1s are available)
    # We need all required to be available.
    # required & available == required
    
    is_available = jnp.all((_ACTION_MASKS & board) == _ACTION_MASKS, axis=1)
    
    # 2. Check Sum
    is_sum_match = (_ACTION_SUMS == target_sum)
    
    return is_available & is_sum_match


def _observe(state: State, player_id: Array) -> Array:
    """
    Observation:
    [0-8]: Board (0=Shut, 1=Open)
    [9-14]: Dice Histogram (counts of 1..6)
    """
    # Board
    board_obs = state._board.astype(jnp.float32)
    
    # Dice Histogram
    # state._dice has values 0-5.
    # We want counts of 0, 1, 2, 3, 4, 5
    dice_vals = jnp.arange(6)
    dice_hist = jnp.sum(state._dice[:, None] == dice_vals[None, :], axis=0).astype(jnp.float32)
    
    return jnp.concatenate([board_obs, dice_hist])
