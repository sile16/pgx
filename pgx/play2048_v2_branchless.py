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
2048 variant with branchless slide/merge operations.

Optimization: Replace jax.lax.cond with jnp.where to avoid branch overhead on GPU/TPU.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.int32(0)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((4, 4, 32), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(4, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _is_stochastic: Array = FALSE
    _stochastic_board: Array = jnp.zeros(16, dtype=jnp.int32)
    _last_spawn_pos: Array = jnp.int32(-1)
    _last_spawn_num: Array = jnp.int32(0)
    _board: Array = jnp.zeros(16, jnp.int32)

    @property
    def env_id(self) -> core.EnvId:
        return "2048"


class Play2048V2Branchless(core.StochasticEnv):
    def __init__(self):
        super().__init__()
        self.stochastic_action_probs = jnp.tile(
            jnp.array([0.9 / 16.0, 0.1 / 16.0], dtype=jnp.float32),
            16,
        )

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

    def _step_deterministic(self, state: State, action: Array) -> State:
        return _step_deterministic(state, action)

    def _step_stochastic(self, state: State, action: Array) -> State:
        return _step_stochastic(state, action)

    def chance_outcomes(self, state: State) -> Tuple[Array, Array]:
        base_board_2d = state._stochastic_board.reshape((4, 4))
        flat = base_board_2d.ravel()
        empty_mask = (flat == 0)
        n_empty = empty_mask.sum()
        p_pos = jnp.where(n_empty > 0, 1.0 / n_empty, 0.0)

        valid_pos = empty_mask[:, None]
        base_probs = jnp.array([0.9, 0.1])
        probs_grid = valid_pos * base_probs * p_pos
        probs_flat = probs_grid.ravel()

        outcomes = jnp.arange(32, dtype=jnp.int32)
        return outcomes, probs_flat

    def stochastic_step(self, state: State, action: Array) -> State:
        return self.step_stochastic(state, action)

    def _step_stochastic_random(self, state: State, key: PRNGKey) -> State:
        base = state._stochastic_board
        k1, k2 = jax.random.split(key)
        empty_mask = (base == 0)
        logits = jnp.where(empty_mask, 0.0, -1e9)
        pos = jax.random.categorical(k1, logits)
        is_four = jax.random.bernoulli(k2, 0.1)
        value_idx = is_four.astype(jnp.int32)
        action = 2 * pos + value_idx
        return self._step_stochastic(state, action)

    def step_stochastic_random(self, state: State, key: PRNGKey) -> State:
        state = self._step_stochastic_random(state, key)
        observation = self._observe(state, state.current_player)
        return state.replace(observation=observation)

    @property
    def id(self) -> core.EnvId:
        return "2048"

    @property
    def version(self) -> str:
        return "v2-branchless"

    @property
    def num_players(self) -> int:
        return 1


def _init(rng: PRNGKey) -> State:
    rng1, rng2 = jax.random.split(rng)
    board = _add_random_num(jnp.zeros((4, 4), jnp.int32), rng1)
    board = _add_random_num(board, rng2)
    return State(
        _board=board.ravel(),
        legal_action_mask=_legal_action_mask(board.reshape((4, 4))),
    )


def _step_deterministic(state: State, action: Array) -> State:
    """action: 0(left), 1(up), 2(right), 3(down)"""
    board_2d = state._board.reshape((4, 4))
    board_2d = jax.lax.switch(
        action,
        [
            lambda: board_2d,
            lambda: jnp.rot90(board_2d, 1),
            lambda: jnp.rot90(board_2d, 2),
            lambda: jnp.rot90(board_2d, 3),
        ],
    )
    # Use branchless slide_and_merge
    board_2d, reward = jax.vmap(_slide_and_merge_branchless)(board_2d)
    board_2d = jax.lax.switch(
        action,
        [
            lambda: board_2d,
            lambda: jnp.rot90(board_2d, -1),
            lambda: jnp.rot90(board_2d, -2),
            lambda: jnp.rot90(board_2d, -3),
        ],
    )

    return state.replace(
        _board=board_2d.ravel(),
        rewards=jnp.float32([reward.sum()]),
        legal_action_mask=jnp.zeros(4, dtype=jnp.bool_),
        terminated=FALSE,
        _is_stochastic=TRUE,
        _stochastic_board=board_2d.ravel(),
        _last_spawn_pos=jnp.int32(-1),
        _last_spawn_num=jnp.int32(0),
    )


def _step_stochastic(state: State, action: Array) -> State:
    base_board_2d = state._stochastic_board.reshape((4, 4))
    pos = (action // 2).astype(jnp.int32)
    value_idx = (action % 2).astype(jnp.int32)
    set_num = (value_idx + 1).astype(jnp.int32)

    board_2d = base_board_2d.at[pos // 4, pos % 4].set(set_num)

    legal_action_mask = _legal_action_mask(board_2d)
    terminated = ~legal_action_mask.any()

    return state.replace(
        _board=board_2d.ravel(),
        legal_action_mask=legal_action_mask,
        terminated=terminated,
        _is_stochastic=FALSE,
        _last_spawn_pos=pos,
        _last_spawn_num=set_num,
    )


def _legal_action_mask(board_2d):
    return jax.vmap(_can_slide_left)(
        jnp.array(
            [
                board_2d,
                jnp.rot90(board_2d, 1),
                jnp.rot90(board_2d, 2),
                jnp.rot90(board_2d, 3),
            ]
        )
    )


def _observe(state: State, player_id) -> Array:
    board = state._board
    obs = jnp.zeros((16, 32), dtype=jnp.bool_)
    obs = obs.at[jnp.arange(16), board].set(TRUE)
    obs = obs.at[:, 31].set(state._is_stochastic)
    return obs.reshape((4, 4, 32))


def _add_random_num(board_2d, key):
    key, sub_key = jax.random.split(key)
    pos = jax.random.choice(key, jnp.arange(16), p=(board_2d.ravel() == 0))
    set_num = jax.random.choice(sub_key, jnp.int32([1, 2]), p=jnp.array([0.9, 0.1]))
    board_2d = board_2d.at[pos // 4, pos % 4].set(set_num)
    return board_2d


# =============================================================================
# BRANCHLESS OPTIMIZATIONS
# =============================================================================

def _slide_left_branchless(line):
    """
    Branchless slide: compact non-zero elements to the left.

    Uses argsort with stability to move zeros to the right.
    Example: [0, 2, 0, 4] -> [2, 4, 0, 0]
    """
    # Create sort keys: 0 for non-zero (comes first), 1 for zero
    is_zero = (line == 0).astype(jnp.int32)
    # argsort with stable=True preserves relative order of equal keys
    indices = jnp.argsort(is_zero, stable=True)
    return line[indices]


def _merge_branchless(line):
    """
    Branchless merge: merge adjacent equal non-zero tiles.

    Uses jnp.where instead of jax.lax.cond for GPU efficiency.
    Example: [2, 2, 4, 4] -> [3, 0, 5, 0] (values are log2, so 3 means 8)
    """
    reward = ZERO

    # Check position 0-1
    can_merge_01 = (line[0] != 0) & (line[0] == line[1])
    new_0 = jnp.where(can_merge_01, line[0] + 1, line[0])
    new_1 = jnp.where(can_merge_01, ZERO, line[1])
    reward_01 = jnp.where(can_merge_01, 2 ** (line[0] + 1), ZERO)

    # Check position 1-2 (using potentially updated line[1])
    can_merge_12 = (new_1 != 0) & (new_1 == line[2])
    new_1 = jnp.where(can_merge_12, new_1 + 1, new_1)
    new_2 = jnp.where(can_merge_12, ZERO, line[2])
    reward_12 = jnp.where(can_merge_12, 2 ** (line[2] + 1), ZERO)

    # Check position 2-3 (using potentially updated line[2])
    can_merge_23 = (new_2 != 0) & (new_2 == line[3])
    new_2 = jnp.where(can_merge_23, new_2 + 1, new_2)
    new_3 = jnp.where(can_merge_23, ZERO, line[3])
    reward_23 = jnp.where(can_merge_23, 2 ** (line[3] + 1), ZERO)

    result = jnp.array([new_0, new_1, new_2, new_3], dtype=line.dtype)
    total_reward = reward_01 + reward_12 + reward_23

    return result, total_reward


def _slide_and_merge_branchless(line):
    """
    Branchless slide and merge: [2, 0, 2, 4] -> [4, 4, 0, 0]

    Process: slide -> merge -> slide
    """
    line = _slide_left_branchless(line)
    line, reward = _merge_branchless(line)
    line = _slide_left_branchless(line)
    return line, reward


def _can_slide_left(board_2d):
    """Check if any row can slide left (unchanged from original)."""
    def _can_slide(line):
        can_slide = (line[0] == 0) & (line[1] > 0)
        can_slide |= (line[1] == 0) & (line[2] > 0)
        can_slide |= (line[2] == 0) & (line[3] > 0)
        can_slide |= (line[0] > 0) & (line[0] == line[1])
        can_slide |= (line[1] > 0) & (line[1] == line[2])
        can_slide |= (line[2] > 0) & (line[2] == line[3])
        can_slide &= ~(line == 0).all()
        return can_slide

    can_slide = jax.vmap(_can_slide)(board_2d).any()
    return can_slide


def show(state):
    board = jnp.array([0 if i == 0 else 2**i for i in state._board])
    print(board.reshape((4, 4)))
