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

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.int32(0)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(16, dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(4, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # Stochastic related
    _is_stochastic: Array = FALSE
    # Board after the player's move but before random tile spawn (raveled 4x4).
    # Used to override the internal stochastic spawn via `stochastic_step`.
    _stochastic_board: Array = jnp.zeros(16, dtype=jnp.int32)
    _last_spawn_pos: Array = jnp.int32(-1)  # 0..15
    _last_spawn_num: Array = jnp.int32(0)  # 1 (2) or 2 (4)
    # --- 2048 specific ---
    # 4x4 board
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]
    _board: Array = jnp.zeros(16, jnp.int32)
    #  Board is expressed as a power of 2.
    # e.g.
    # [[ 0,  0,  1,  1],
    #  [ 1,  0,  1,  2],
    #  [ 3,  3,  6,  7],
    #  [ 3,  6,  7,  9]]
    # means
    # [[ 0,  0,  2,  2],
    #  [ 2,  0,  2,  4],
    #  [ 8,  8, 64,128],
    #  [ 8, 64,128,512]]

    @property
    def env_id(self) -> core.EnvId:
        return "2048"


class Play2048(core.Env):
    def __init__(self):
        super().__init__()
        # (pos, value) where pos=0..15, value in {2,4}.
        # action = 2 * pos + value_idx, value_idx: 0->2, 1->4.
        self.stochastic_action_probs = jnp.tile(
            jnp.array([0.9 / 16.0, 0.1 / 16.0], dtype=jnp.float32),
            16,
        )

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

    def stochastic_step(self, state: State, action: Array) -> State:
        return _stochastic_step(state, action)

    @property
    def id(self) -> core.EnvId:
        return "2048"

    @property
    def version(self) -> str:
        return "v2"

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
    )  # type:ignore


def _step(state: State, action, key):
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
    board_2d, reward = jax.vmap(_slide_and_merge)(board_2d)
    board_2d = jax.lax.switch(
        action,
        [
            lambda: board_2d,
            lambda: jnp.rot90(board_2d, -1),
            lambda: jnp.rot90(board_2d, -2),
            lambda: jnp.rot90(board_2d, -3),
        ],
    )

    board_before_spawn = board_2d
    board_2d, (spawn_pos, spawn_num) = _add_random_num_with_info(board_2d, key)

    legal_action_mask = _legal_action_mask(board_2d)
    return state.replace(  # type:ignore
        _board=board_2d.ravel(),
        rewards=jnp.float32([reward.sum()]),
        legal_action_mask=legal_action_mask,
        terminated=~legal_action_mask.any(),
        _is_stochastic=TRUE,
        _stochastic_board=board_before_spawn.ravel(),
        _last_spawn_pos=spawn_pos,
        _last_spawn_num=spawn_num,
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
    obs = jnp.zeros((16, 31), dtype=jnp.bool_)
    obs = jax.lax.fori_loop(0, 16, lambda i, obs: obs.at[i, state._board[i]].set(TRUE), obs)
    return obs.reshape((4, 4, 31))


def _add_random_num(board_2d, key):
    """Add 2 or 4 to the empty space on the board.
    2 appears 90% of the time, and 4 appears 10% of the time.
    cf. https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js#L71
    """
    key, sub_key = jax.random.split(key)
    pos = jax.random.choice(key, jnp.arange(16), p=(board_2d.ravel() == 0))
    set_num = jax.random.choice(sub_key, jnp.int32([1, 2]), p=jnp.array([0.9, 0.1]))
    board_2d = board_2d.at[pos // 4, pos % 4].set(set_num)
    return board_2d


def _add_random_num_with_info(board_2d, key):
    """Like `_add_random_num`, but also returns (pos, set_num)."""
    key, sub_key = jax.random.split(key)
    pos = jax.random.choice(key, jnp.arange(16), p=(board_2d.ravel() == 0))
    set_num = jax.random.choice(sub_key, jnp.int32([1, 2]), p=jnp.array([0.9, 0.1]))
    board_2d = board_2d.at[pos // 4, pos % 4].set(set_num)
    return board_2d, (pos.astype(jnp.int32), set_num.astype(jnp.int32))


def _stochastic_step(state: State, action: Array) -> State:
    """Override the internally sampled tile spawn of the previous step.

    action = 2 * pos + value_idx, where pos=0..15 and value_idx: 0->2, 1->4.
    """
    base_board_2d = state._stochastic_board.reshape((4, 4))
    pos = (action // 2).astype(jnp.int32)
    value_idx = (action % 2).astype(jnp.int32)
    set_num = (value_idx + 1).astype(jnp.int32)  # 1 (2) or 2 (4)

    base_flat = base_board_2d.ravel()
    can_place = base_flat[pos] == 0

    proposed_board_2d = base_board_2d.at[pos // 4, pos % 4].set(set_num)
    proposed_mask = _legal_action_mask(proposed_board_2d)
    proposed_terminated = ~proposed_mask.any()
    proposed_mask = jax.lax.select(proposed_terminated, jnp.ones_like(proposed_mask), proposed_mask)

    return state.replace(  # type:ignore
        _board=jnp.where(can_place, proposed_board_2d.ravel(), state._board),
        legal_action_mask=jnp.where(can_place, proposed_mask, state.legal_action_mask),
        terminated=jnp.where(can_place, proposed_terminated, state.terminated),
        _is_stochastic=jnp.where(can_place, FALSE, state._is_stochastic),
        _stochastic_board=jnp.where(
            can_place,
            jnp.zeros_like(state._stochastic_board),
            state._stochastic_board,
        ),
        _last_spawn_pos=jnp.where(can_place, pos, state._last_spawn_pos),
        _last_spawn_num=jnp.where(can_place, set_num, state._last_spawn_num),
    )


def _slide_and_merge(line):
    """[2 2 2 2] -> [4 4 0 0]"""
    line = _slide_left(line)
    line, reward = _merge(line)
    line = _slide_left(line)
    return line, reward


def _merge(line):
    """[2 2 2 2] -> [4 0 4 0]"""
    line, reward = jax.lax.cond(
        (line[0] != 0) & (line[0] == line[1]),
        lambda: (
            line.at[0].set(line[0] + 1).at[1].set(ZERO),
            2 ** (line[0] + 1),
        ),
        lambda: (line, ZERO),
    )
    line, reward = jax.lax.cond(
        (line[1] != 0) & (line[1] == line[2]),
        lambda: (
            line.at[1].set(line[1] + 1).at[2].set(ZERO),
            reward + 2 ** (line[1] + 1),
        ),
        lambda: (line, reward),
    )
    line, reward = jax.lax.cond(
        (line[2] != 0) & (line[2] == line[3]),
        lambda: (
            line.at[2].set(line[2] + 1).at[3].set(ZERO),
            reward + 2 ** (line[2] + 1),
        ),
        lambda: (line, reward),
    )
    return line, reward


def _slide_left(line):
    """[0 2 0 2] -> [2 2 0 0]"""
    line = jax.lax.cond(
        (line[2] == 0),
        lambda: line.at[2:].set(jnp.roll(line[2:], -1)),
        lambda: line,
    )
    line = jax.lax.cond(
        (line[1] == 0),
        lambda: line.at[1:].set(jnp.roll(line[1:], -1)),
        lambda: line,
    )
    line = jax.lax.cond(
        (line[0] == 0),
        lambda: jnp.roll(line, -1),
        lambda: line,
    )
    return line


def _can_slide_left(board_2d):
    def _can_slide(line):
        """Judge if it can be moved to the left."""
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


# only for debug
def show(state):
    board = jnp.array([0 if i == 0 else 2**i for i in state._board])
    print(board.reshape((4, 4)))


def stochastic_action_to_str(action: Array) -> str:
    pos = int(action) // 2
    value_idx = int(action) % 2
    tile = 2 if value_idx == 0 else 4
    r, c = divmod(pos, 4)
    return f"Spawned: {tile} at ({r},{c})"
