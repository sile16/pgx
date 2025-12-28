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
Connect Four with optimized operations.

Optimizations:
1. Precomputed win pattern indices for faster vectorized checking
2. Efficient column height tracking
3. Branchless operations where possible

This version maintains compatibility with the original implementation
while optimizing for GPU execution.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


class GameState(NamedTuple):
    """Compatibility wrapper for tests that access _x.board and _x.winner."""
    color: Array = jnp.int32(0)
    board: Array = -jnp.ones(42, jnp.int32)
    winner: Array = jnp.int32(-1)


# Precomputed win pattern indices (same as original, but hoisted to module level)
def _make_win_cache():
    idx = []
    # Vertical
    for i in range(3):
        for j in range(7):
            a = i * 7 + j
            idx.append([a, a + 7, a + 14, a + 21])
    # Horizontal
    for i in range(6):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 1, a + 2, a + 3])
    # Diagonal (ascending)
    for i in range(3):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 8, a + 16, a + 24])
    # Diagonal (descending)
    for i in range(3):
        for j in range(3, 7):
            a = i * 7 + j
            idx.append([a, a + 6, a + 12, a + 18])
    return jnp.int32(idx)


_IDX = _make_win_cache()  # Shape: (69, 4)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((6, 7, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(7, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # Board representation: -1 = empty, 0 = player 0, 1 = player 1
    _board: Array = -jnp.ones(42, jnp.int32)
    _color: Array = jnp.int32(0)  # Current player's color
    _winner: Array = jnp.int32(-1)
    # Track column heights for O(1) placement (count of pieces in each column)
    _heights: Array = jnp.zeros(7, jnp.int32)

    @property
    def env_id(self) -> core.EnvId:
        return "connect_four"

    @property
    def _x(self) -> GameState:
        """Compatibility property for tests that access _x.board and _x.winner."""
        return GameState(color=self._color, board=self._board, winner=self._winner)


class ConnectFourV2Bitboard(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        return State(current_player=current_player)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)

        board = state._board
        color = state._color
        heights = state._heights

        # Calculate position: row = 5 - heights[action], col = action
        row = 5 - heights[action]
        pos = row * 7 + action

        # Place piece
        new_board = board.at[pos].set(color)
        new_heights = heights.at[action].add(1)

        # Check for win using vectorized pattern matching
        won = _check_win_fast(new_board, color)
        winner = jnp.where(won, color, jnp.int32(-1))

        # Switch color
        new_color = 1 - color
        new_current_player = 1 - state.current_player

        # Compute legal actions (columns not full)
        legal_action_mask = new_heights < 6

        # Check for terminal (win or draw)
        is_draw = ~legal_action_mask.any()
        terminated = won | is_draw

        # Compute rewards
        rewards = jnp.where(
            won,
            jnp.float32([-1.0, -1.0]).at[color].set(1.0),
            jnp.zeros(2, jnp.float32),
        )
        # Flip rewards to match current_player perspective
        should_flip = new_current_player != new_color
        rewards = jnp.where(should_flip, jnp.flip(rewards), rewards)
        rewards = jnp.where(terminated, rewards, jnp.zeros(2, jnp.float32))

        return state.replace(
            current_player=new_current_player,
            _board=new_board,
            _color=new_color,
            _winner=winner,
            _heights=new_heights,
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        curr_color = state._color
        my_color = jax.lax.select(
            player_id == state.current_player, curr_color, 1 - curr_color
        )
        return _observe(state._board, my_color)

    @property
    def id(self) -> core.EnvId:
        return "connect_four"

    @property
    def version(self) -> str:
        return "v2-bitboard"

    @property
    def num_players(self) -> int:
        return 2


def _check_win_fast(board: Array, color: Array) -> Array:
    """Check if the given color has won using vectorized pattern matching.

    Uses precomputed indices to check all 69 possible winning patterns at once.
    """
    # Check all patterns: board[_IDX] == color gives (69, 4) boolean array
    # .all(axis=1) gives (69,) boolean array - True if pattern is complete
    # .any() gives scalar boolean - True if any pattern is complete
    patterns = board[_IDX]  # Shape: (69, 4)
    matches = (patterns == color).all(axis=1)  # Shape: (69,)
    return matches.any()


def _observe(board: Array, my_color: Array) -> Array:
    """Convert board to observation tensor (6, 7, 2).

    Channel 0: my pieces
    Channel 1: opponent pieces
    """
    board_2d = board.reshape(6, 7)
    my_pieces = board_2d == my_color
    opp_pieces = board_2d == (1 - my_color)
    return jnp.stack([my_pieces, opp_pieces], axis=-1)
