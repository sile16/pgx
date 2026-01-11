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
Backgammon environment with 2-move actions (26 x 26).

Action encoding:
  - Each action encodes two source selections: (src for die1, src for die2)
  - Each src is one of: pass(0), bar(1), points 1-24 (2-25)
"""

from functools import partial
from typing import Optional, Tuple
import dataclasses

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

_SRC_INDICES = jnp.arange(26, dtype=jnp.int32)
_NO_OP_ACTION_MASK = jnp.zeros(26 * 26, dtype=jnp.bool_).at[0].set(True)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(34, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    _is_stochastic: Array = jnp.array(True, dtype=jnp.bool_)
    legal_action_mask: Array = jnp.zeros(26 * 26, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _board: Array = jnp.zeros(28, dtype=jnp.int32)
    _dice: Array = jnp.zeros(2, dtype=jnp.int32)
    _remaining_actions: Array = jnp.int32(0)
    _turn: Array = jnp.int32(1)

    @property
    def env_id(self) -> core.EnvId:
        return "backgammon"

    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)


class Backgammon2P(core.StochasticEnv):
    def __init__(self, short_game: bool = False):
        super().__init__()
        self.init_board_fn = _make_init_board_short if short_game else _make_init_board

    def step(self, state: core.State, action: Array, key: Optional[Array] = None) -> core.State:
        return super().step(state, action, key)

    def _init(self, key: PRNGKey) -> State:
        state = _init(key)
        state = state.replace(_board=self.init_board_fn())
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)
        return super()._step(state, action, key)

    def _observe(self, state: core.State, player_id: Optional[Array] = None) -> Array:
        assert isinstance(state, State)
        return _observe_fast(state)

    def _step_deterministic(self, state: State, action: Array) -> State:
        return _decision_step(state, action)

    def _step_stochastic(self, state: State, action: Array) -> State:
        dice_idx = action
        dice = _normalize_dice(_STOCHASTIC_DICE_MAPPING[dice_idx])
        remaining_actions = _remaining_actions_from_dice(dice)
        legal_action_mask = _legal_action_mask(state._board, dice)
        return state.replace(
            _dice=dice,
            _remaining_actions=remaining_actions,
            legal_action_mask=legal_action_mask,
            _is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )

    def stochastic_action_probs(self, state: State) -> Array:
        return _STOCHASTIC_ACTION_PROBS

    @property
    def static_stochastic_action_probs(self) -> Array:
        return _STOCHASTIC_ACTION_PROBS

    def chance_outcomes(self, state: State) -> Tuple[Array, Array]:
        outcomes = jnp.arange(self.num_stochastic_actions, dtype=jnp.int32)
        return outcomes, self.stochastic_action_probs(state)

    @property
    def id(self) -> core.EnvId:
        return "backgammon"

    @property
    def version(self) -> str:
        return "2p"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_stochastic_actions(self) -> int:
        return int(self.static_stochastic_action_probs.shape[0])

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0

    def stochastic_step(self, state: State, action: Array) -> State:
        return self.step_stochastic(state, action)

    def _set_dice(self, state: State, dice: Array) -> State:
        dice = _normalize_dice(dice)
        remaining_actions = _remaining_actions_from_dice(dice)
        legal_action_mask = _legal_action_mask(state._board, dice)
        return state.replace(
            _dice=dice,
            _remaining_actions=remaining_actions,
            legal_action_mask=legal_action_mask,
            _is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )

    def set_dice(self, state: State, dice: Array) -> State:
        state = self._set_dice(state, dice)
        observation = self._observe(state, state.current_player)
        return state.replace(observation=observation)


# =============================================================================
# Observation
# =============================================================================

def _observe_fast(state: State) -> Array:
    board = state._board.astype(jnp.float32)
    scaled_board = board / 15.0
    scaled_dice = jax.lax.cond(
        state._is_stochastic,
        lambda: jnp.zeros(6, dtype=jnp.float32),
        lambda: _remaining_dice_count(state._dice, state._remaining_actions).astype(jnp.float32) / 4.0,
    )
    return jnp.concatenate([scaled_board, scaled_dice])


def _remaining_dice_count(dice: Array, remaining_actions: Array) -> Array:
    die_values = jnp.arange(6, dtype=jnp.int32)
    is_doubles = dice[0] == dice[1]
    remaining_dice = jnp.where(is_doubles, remaining_actions * 2, 2)

    def count_die(die_val):
        return jnp.where(
            (dice[0] == die_val) & is_doubles,
            remaining_dice,
            (dice == die_val).sum(),
        )

    return jax.vmap(count_die)(die_values).astype(jnp.int32)


# =============================================================================
# Core game logic
# =============================================================================

def _init(rng: PRNGKey) -> State:
    rng1, rng2 = jax.random.split(rng, num=2)
    current_player: Array = jax.random.bernoulli(rng1).astype(jnp.int32)
    board: Array = _make_init_board()
    terminated: Array = FALSE
    dice: Array = _roll_init_dice(rng2)
    remaining_actions = _remaining_actions_from_dice(dice)
    turn: Array = _init_turn(dice)
    legal_action_mask: Array = _legal_action_mask(board, dice)
    state = State(
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _dice=dice,
        _remaining_actions=remaining_actions,
        _turn=turn,
        legal_action_mask=legal_action_mask,
        _is_stochastic=jnp.array(True, dtype=jnp.bool_),
    )
    return state


def _decision_step(state: State, action: Array) -> State:
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state._board),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state, action),
    )


def _winning_step(state: State) -> State:
    win_score = _calc_win_score(state._board)
    winner = state.current_player
    loser = 1 - winner
    reward = jnp.ones_like(state.rewards)
    reward = reward.at[winner].set(win_score)
    reward = reward.at[loser].set(-win_score)
    state = state.replace(terminated=TRUE)
    return state.replace(rewards=reward)


def _no_winning_step(state: State, action: Array, key=None) -> State:
    return jax.lax.cond(
        (_is_turn_end(state) | (action == 0)),
        lambda: _change_turn(state, key),
        lambda: state,
    )


def _update_by_action(state: State, action: Array) -> State:
    src1_idx, src2_idx = _decode_action(action)
    die1 = state._dice[0] + 1
    die2 = state._dice[1] + 1
    board_after_move = _apply_action(state._board, src1_idx, src2_idx, die1, die2)
    remaining_actions_next = state._remaining_actions - 1
    legal_action_mask_next = _legal_action_mask(board_after_move, state._dice)
    no_moves_left = _is_no_op_mask(legal_action_mask_next)
    remaining_actions = jnp.where(no_moves_left, 0, remaining_actions_next)
    legal_action_mask = jax.lax.cond(
        remaining_actions > 0,
        lambda: legal_action_mask_next,
        lambda: jnp.zeros(26 * 26, dtype=jnp.bool_),
    )
    return state.replace(
        _board=board_after_move,
        _remaining_actions=remaining_actions,
        legal_action_mask=legal_action_mask,
    )


def _is_turn_end(state: State) -> bool:
    return state._remaining_actions == 0


def _change_turn(state: State, key=None) -> State:
    board: Array = _flip_board(state._board)
    turn: Array = (state._turn + 1) % 2
    current_player: Array = (state.current_player + 1) % 2
    terminated: Array = state.terminated
    dice = jnp.zeros(2, dtype=jnp.int32)
    remaining_actions = jnp.int32(0)
    legal_action_mask = jnp.zeros(26 * 26, dtype=jnp.bool_)
    return state.replace(
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _turn=turn,
        _dice=dice,
        _remaining_actions=remaining_actions,
        legal_action_mask=legal_action_mask,
        _is_stochastic=jnp.array(True, dtype=jnp.bool_),
    )


def _flip_board(board):
    _board = board
    board = board.at[:24].set(jnp.flip(_board[:24]))
    board = board.at[24:26].set(jnp.flip(_board[24:26]))
    board = board.at[26:28].set(jnp.flip(_board[26:28]))
    return -1 * board


def _make_init_board() -> Array:
    return jnp.array(
        [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0],
        dtype=jnp.int32,
    )


def _make_init_board_short() -> Array:
    return jnp.array(
        [0, -1, -3, 0, 2, -3, 0, -3, -2, 0, 0, -1, 1, 0, 0, 2, 3, 0, 3, -2, 0, 3, 1, 0, 0, 0, 0, 0],
        dtype=jnp.int32,
    )


def _roll_init_dice(rng: PRNGKey) -> Array:
    init_dice_pattern = jnp.array(
        [
            [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5],
            [2, 0], [2, 1], [2, 3], [2, 4], [2, 5], [3, 0], [3, 1], [3, 2], [3, 4], [3, 5],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4],
        ],
        dtype=jnp.int32,
    )
    return jax.random.choice(rng, init_dice_pattern)


def _normalize_dice(dice: Array) -> Array:
    is_doubles = dice[0] == dice[1]
    sorted_dice = jnp.sort(dice)
    return jnp.where(is_doubles, dice, sorted_dice)


def _remaining_actions_from_dice(dice: Array) -> Array:
    is_doubles = dice[0] == dice[1]
    return jnp.where(is_doubles, jnp.int32(2), jnp.int32(1))


def _init_turn(dice: Array) -> Array:
    diff = dice[1] - dice[0]
    return jnp.int32(diff > 0)


def _is_all_off(board: Array) -> bool:
    return board[_off_idx()] == 15


def _calc_win_score(board: Array) -> int:
    g = _is_gammon(board)
    return 1 + g + (g & _remains_at_inner(board))


def _is_gammon(board: Array) -> bool:
    return board[_off_idx() + 1] == 0


def _remains_at_inner(board: Array) -> bool:
    return jnp.take(board, _home_board()).sum() != 0


def _home_board() -> Array:
    return jnp.arange(18, 24, dtype=jnp.int32)


def _off_idx() -> int:
    return 26


def _bar_idx() -> int:
    return 24


def _decode_action(action: Array):
    return action // 26, action % 26


def _src_from_idx(src_idx: Array) -> Array:
    return jnp.where(src_idx == 0, -2, jnp.where(src_idx == 1, 24, src_idx - 2))


def _tgt_from_src_die(src: Array, die: Array) -> Array:
    tgt_from_bar = die - 1
    tgt_from_board = src + die
    tgt_normal = jnp.where((tgt_from_board >= 0) & (tgt_from_board <= 23), tgt_from_board, 26)
    tgt = jnp.where(src >= 24, tgt_from_bar, tgt_normal)
    tgt = jnp.where(src == -2, -2, tgt)
    return tgt


def _is_move_legal(board: Array, src_idx: Array, die: Array) -> bool:
    src = _src_from_idx(src_idx)
    tgt = _tgt_from_src_die(src, die)

    src_safe = jnp.clip(src, 0, 27)
    tgt_safe = jnp.clip(tgt, 0, 27)

    is_to_point = (tgt >= 0) & (tgt <= 23) & (src >= 0)
    is_to_off = (tgt == 26) & (src >= 0)

    src_has_checker = board[src_safe] >= 1
    tgt_is_open = board[tgt_safe] >= -1
    bar_clear = board[24] == 0
    from_bar = src >= 24

    point_legal = src_has_checker & tgt_is_open & (from_bar | bar_clear)

    home_board_indices = jnp.arange(18, 24)
    on_home_board = jnp.sum(jnp.where(board[home_board_indices] > 0, board[home_board_indices], 0))
    off_count = board[26]
    all_on_home = (15 - off_count) == on_home_board

    dist = 24 - src
    has_checker = board[:24] > 0
    rear_pos = jnp.where(has_checker, jnp.arange(24), 100)
    rear_idx = jnp.min(rear_pos)
    rear_dist = 24 - rear_idx

    exact_roll = dist == die
    highest_and_overbear = (rear_dist <= die) & (rear_dist == dist)
    off_legal = src_has_checker & all_on_home & (exact_roll | highest_and_overbear)

    is_no_op = src == -2
    result = jnp.where(is_to_point, point_legal, jnp.where(is_to_off, off_legal, False))
    return jnp.where(is_no_op, False, result)


def _apply_move(board: Array, src_idx: Array, die: Array) -> Array:
    src = _src_from_idx(src_idx)
    tgt = _tgt_from_src_die(src, die)

    def do_move(b):
        b = b.at[_bar_idx() + 1].add(-1 * (b[tgt] == -1))
        b = b.at[src].add(-1)
        b = b.at[tgt].add(1 + (b[tgt] == -1))
        return b

    return jax.lax.cond(src_idx == 0, lambda: board, lambda: do_move(board))


def _apply_move_if_legal(board: Array, src_idx: Array, die: Array) -> Array:
    return jax.lax.cond(
        _is_move_legal(board, src_idx, die),
        lambda: _apply_move(board, src_idx, die),
        lambda: board,
    )


def _any_legal_move(board: Array, die: Array) -> Array:
    is_legal = jax.vmap(lambda src_idx: _is_move_legal(board, src_idx, die))(_SRC_INDICES)
    return is_legal.any()


def _can_play_sequence(board: Array, die_first: Array, die_second: Array) -> Array:
    is_legal_first = jax.vmap(lambda src_idx: _is_move_legal(board, src_idx, die_first))(_SRC_INDICES)
    next_boards = jax.vmap(lambda src_idx: _apply_move_if_legal(board, src_idx, die_first))(_SRC_INDICES)
    can_play_second = jax.vmap(lambda next_board: _any_legal_move(next_board, die_second))(next_boards)
    return (is_legal_first & can_play_second).any()


def _can_play_two_moves(board: Array, die1: Array, die2: Array) -> Array:
    return _can_play_sequence(board, die1, die2) | _can_play_sequence(board, die2, die1)


def _is_action_two_moves(board: Array, src1_idx: Array, src2_idx: Array, die1: Array, die2: Array) -> Array:
    def seq(src_first, die_first, src_second, die_second):
        legal_first = _is_move_legal(board, src_first, die_first)
        board_after = _apply_move(board, src_first, die_first)
        legal_second = _is_move_legal(board_after, src_second, die_second)
        return legal_first & legal_second

    seq1 = seq(src1_idx, die1, src2_idx, die2)
    seq2 = seq(src2_idx, die2, src1_idx, die1)
    return seq1 | seq2


def _apply_action(board: Array, src1_idx: Array, src2_idx: Array, die1: Array, die2: Array) -> Array:
    is_pass1 = src1_idx == 0
    is_pass2 = src2_idx == 0

    def apply_single(src_idx, die_val):
        return _apply_move(board, src_idx, die_val)

    def apply_double():
        def seq(src_first, die_first, src_second, die_second):
            board_after = _apply_move(board, src_first, die_first)
            return _apply_move(board_after, src_second, die_second)

        seq1_legal = _is_move_legal(board, src1_idx, die1) & _is_move_legal(
            _apply_move(board, src1_idx, die1), src2_idx, die2
        )
        return jax.lax.cond(
            seq1_legal,
            lambda: seq(src1_idx, die1, src2_idx, die2),
            lambda: seq(src2_idx, die2, src1_idx, die1),
        )

    return jax.lax.cond(
        is_pass1 & is_pass2,
        lambda: board,
        lambda: jax.lax.cond(
            is_pass1,
            lambda: apply_single(src2_idx, die2),
            lambda: jax.lax.cond(
                is_pass2,
                lambda: apply_single(src1_idx, die1),
                apply_double,
            ),
        ),
    )


def _legal_action_mask_nondoubles(board: Array, die1: Array, die2: Array) -> Array:
    can_play_d1 = _any_legal_move(board, die1)
    can_play_d2 = _any_legal_move(board, die2)
    can_play_both = _can_play_two_moves(board, die1, die2)

    def action_legal(action):
        src1_idx, src2_idx = _decode_action(action)
        move1 = _is_move_legal(board, src1_idx, die1)
        move2 = _is_move_legal(board, src2_idx, die2)
        two_moves = _is_action_two_moves(board, src1_idx, src2_idx, die1, die2)

        def both():
            return two_moves

        def single():
            def use_die2():
                return (src1_idx == 0) & move2

            def use_die1():
                return (src2_idx == 0) & move1

            return jax.lax.cond(can_play_d2, use_die2, use_die1)

        def none():
            return (src1_idx == 0) & (src2_idx == 0)

        return jax.lax.cond(
            can_play_both,
            both,
            lambda: jax.lax.cond((can_play_d1 | can_play_d2), single, none),
        )

    actions = jnp.arange(26 * 26, dtype=jnp.int32)
    mask = jax.vmap(action_legal)(actions)
    return jax.lax.cond(mask.any(), lambda: mask, lambda: _NO_OP_ACTION_MASK)


def _legal_action_mask_doubles(board: Array, die: Array) -> Array:
    can_play_one = _any_legal_move(board, die)
    can_play_two = _can_play_two_moves(board, die, die)

    def action_legal(action):
        src1_idx, src2_idx = _decode_action(action)
        move1 = _is_move_legal(board, src1_idx, die)
        move2 = _is_move_legal(board, src2_idx, die)
        two_moves = _is_action_two_moves(board, src1_idx, src2_idx, die, die)

        def both():
            return two_moves

        def single():
            return ((src1_idx == 0) & move2) | ((src2_idx == 0) & move1)

        def none():
            return (src1_idx == 0) & (src2_idx == 0)

        return jax.lax.cond(
            can_play_two,
            both,
            lambda: jax.lax.cond(can_play_one, single, none),
        )

    actions = jnp.arange(26 * 26, dtype=jnp.int32)
    mask = jax.vmap(action_legal)(actions)
    return jax.lax.cond(mask.any(), lambda: mask, lambda: _NO_OP_ACTION_MASK)


def _legal_action_mask(board: Array, dice: Array) -> Array:
    is_doubles = dice[0] == dice[1]
    die1 = dice[0] + 1
    die2 = dice[1] + 1
    return jax.lax.cond(
        is_doubles,
        lambda: _legal_action_mask_doubles(board, die1),
        lambda: _legal_action_mask_nondoubles(board, die1, die2),
    )


def _is_no_op_mask(mask: Array) -> Array:
    return jnp.all(mask == _NO_OP_ACTION_MASK)


# Probability distributions
_STOCHASTIC_ACTION_PROBS = jnp.array(
    [
        1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
        2 / 36, 2 / 36, 2 / 36, 2 / 36, 2 / 36,
        2 / 36, 2 / 36, 2 / 36, 2 / 36,
        2 / 36, 2 / 36, 2 / 36,
        2 / 36, 2 / 36,
        2 / 36,
    ],
    dtype=jnp.float32,
)

_STOCHASTIC_DICE_MAPPING = jnp.array(
    [
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [3, 4], [3, 5],
        [4, 5],
    ],
    dtype=jnp.int32,
)


def action_to_str(action) -> str:
    action_int = int(action)
    src1_idx, src2_idx = _decode_action(jnp.int32(action_int))

    def src_to_str(src_idx):
        src = int(_src_from_idx(jnp.int32(src_idx)))
        if src == -2:
            return "Pass"
        if src >= 24:
            return "Bar"
        return str(src + 1)

    return f"{src_to_str(src1_idx)} | {src_to_str(src2_idx)}"
