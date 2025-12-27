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
Backgammon variant with ALL optimizations combined.

Optimizations:
1. Fast/minimal observation (skip heuristics) - 34 elements instead of 86
2. Branchless operations using jnp.where
3. Vectorized pip counts using dot products
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

# Hoisted constants
_SRC_INDICES = jnp.arange(26, dtype=jnp.int32)
_NO_OP_MASK = jnp.zeros(26 * 6, dtype=jnp.bool_).at[0:6].set(True)


def _build_lookup_tables():
    actions = jnp.arange(26 * 6, dtype=jnp.int32)
    src_raw = actions // 6
    src = jnp.where(src_raw == 1, 24, jnp.where(src_raw == 0, -2, src_raw - 2))
    die = (actions % 6) + 1
    tgt_from_bar = die - 1
    tgt_from_board = src + die
    tgt_normal = jnp.where((tgt_from_board >= 0) & (tgt_from_board <= 23), tgt_from_board, 26)
    tgt = jnp.where(src >= 24, tgt_from_bar, tgt_normal)
    tgt = jnp.where(src == -2, -2, tgt)
    return src.astype(jnp.int32), die.astype(jnp.int32), tgt.astype(jnp.int32)


_ACTION_SRC_LOOKUP, _ACTION_DIE_LOOKUP, _ACTION_TGT_LOOKUP = _build_lookup_tables()


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    # Reduced observation: 34 elements (board + dice only)
    observation: Array = jnp.zeros(34, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    _is_stochastic: Array = jnp.array(True, dtype=jnp.bool_)
    legal_action_mask: Array = jnp.zeros(6 * 26, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _board: Array = jnp.zeros(28, dtype=jnp.int32)
    _dice: Array = jnp.zeros(2, dtype=jnp.int32)
    _playable_dice: Array = jnp.zeros(4, dtype=jnp.int32)
    _played_dice_num: Array = jnp.int32(0)
    _turn: Array = jnp.int32(1)

    @property
    def env_id(self) -> core.EnvId:
        return "backgammon"

    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)


class BackgammonV2All(core.StochasticEnv):
    def __init__(self, simple_doubles: bool = False, short_game: bool = False):
        super().__init__()
        self.simple_doubles = simple_doubles
        self.init_board_fn = _make_init_board_short if short_game else _make_init_board
        self.stochastic_action_probs = (
            _STOCHASTIC_SIMPLE_DOUBLES_ACTION_PROBS if simple_doubles
            else _STOCHASTIC_ACTION_PROBS
        )

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
        dice = _STOCHASTIC_DICE_MAPPING[dice_idx]
        playable_dice: Array = _set_playable_dice(dice)
        played_dice_num: Array = jnp.int32(0)
        legal_action_mask: Array = _legal_action_mask(state._board, playable_dice, dice, played_dice_num)
        return state.replace(
            _dice=dice,
            _playable_dice=playable_dice,
            _played_dice_num=played_dice_num,
            legal_action_mask=legal_action_mask,
            _is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )

    def chance_outcomes(self, state: State) -> Tuple[Array, Array]:
        outcomes = jnp.arange(len(self.stochastic_action_probs), dtype=jnp.int32)
        return outcomes, self.stochastic_action_probs

    @property
    def id(self) -> core.EnvId:
        return "backgammon"

    @property
    def version(self) -> str:
        return "v2-all"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0


# =============================================================================
# FAST OBSERVATION (minimal, no heuristics)
# =============================================================================

def _observe_fast(state: State) -> Array:
    """
    Fast observation: just board and dice, no heuristics.
    34 elements instead of 86.
    """
    board = state._board
    board_f = board.astype(jnp.float32)
    scaled_board = board_f / 15.0

    scaled_dice = jax.lax.cond(
        state._is_stochastic,
        lambda: jnp.zeros(6, dtype=jnp.float32),
        lambda: _to_playable_dice_count(state._playable_dice).astype(jnp.float32) / 4.0
    )

    return jnp.concatenate([scaled_board, scaled_dice])


def _to_playable_dice_count(playable_dice: Array) -> Array:
    valid_mask = playable_dice != -1
    die_values = jnp.arange(6, dtype=jnp.int32)

    def count_die(die_val):
        return ((playable_dice == die_val) & valid_mask).sum()

    return jax.vmap(count_die)(die_values).astype(jnp.int32)


# =============================================================================
# BRANCHLESS OPERATIONS
# =============================================================================

def _is_action_legal_branchless(board: Array, action: Array) -> bool:
    """Branchless action legality check using jnp.where."""
    src, die, tgt = _decompose_action(action)

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

    result = jnp.where(is_to_point, point_legal, jnp.where(is_to_off, off_legal, False))
    return result


def _update_by_action_branchless(state: State, action: Array) -> State:
    """Branchless action update."""
    is_no_op = action // 6 == 0

    board_after_move: Array = _move(state._board, action)
    played_dice_num_new: Array = jnp.int32(state._played_dice_num + 1)
    playable_dice_new: Array = _update_playable_dice(state._playable_dice, state._played_dice_num, state._dice, action)
    legal_action_mask_new: Array = _legal_action_mask(board_after_move, playable_dice_new, state._dice, played_dice_num_new)

    board = jnp.where(is_no_op, state._board, board_after_move)
    playable_dice = jnp.where(is_no_op, state._playable_dice, playable_dice_new)
    played_dice_num = jnp.where(is_no_op, state._played_dice_num, played_dice_num_new)
    legal_action_mask = jnp.where(is_no_op, state.legal_action_mask, legal_action_mask_new)

    return state.replace(
        _board=board,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


def _legal_action_mask_for_single_die_branchless(board: Array, die) -> Array:
    """Branchless single die legal action mask."""
    actions = _SRC_INDICES * 6 + die
    is_legal = jax.vmap(_is_action_legal_branchless, in_axes=(None, 0))(board, actions)
    mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
    result = mask.at[actions].set(is_legal)
    is_valid_die = die != -1
    return jnp.where(is_valid_die, result, jnp.zeros(26 * 6, dtype=jnp.bool_))


# =============================================================================
# Core game logic
# =============================================================================

def _init(rng: PRNGKey) -> State:
    rng1, rng2 = jax.random.split(rng, num=2)
    current_player: Array = jax.random.bernoulli(rng1).astype(jnp.int32)
    board: Array = _make_init_board()
    terminated: Array = FALSE
    dice: Array = _roll_init_dice(rng2)
    playable_dice: Array = _set_playable_dice(dice)
    played_dice_num: Array = jnp.int32(0)
    turn: Array = _init_turn(dice)
    legal_action_mask: Array = _legal_action_mask(board, playable_dice, dice, played_dice_num)
    state = State(
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        _turn=turn,
        legal_action_mask=legal_action_mask,
        _is_stochastic=jnp.array(True, dtype=jnp.bool_)
    )
    return state


def _decision_step(state: State, action: Array) -> State:
    state = _update_by_action_branchless(state, action)
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
        (_is_turn_end(state) | (action // 6 == 0)),
        lambda: _change_turn(state, key),
        lambda: state,
    )


def _flip_board(board):
    _board = board
    board = board.at[:24].set(jnp.flip(_board[:24]))
    board = board.at[24:26].set(jnp.flip(_board[24:26]))
    board = board.at[26:28].set(jnp.flip(_board[26:28]))
    return -1 * board


def _make_init_board() -> Array:
    return jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)


def _make_init_board_short() -> Array:
    return jnp.array([0, -1, -3, 0, 2, -3, 0, -3, -2, 0, 0, -1, 1, 0, 0, 2, 3, 0, 3, -2, 0, 3, 1, 0, 0, 0, 0, 0], dtype=jnp.int32)


def _is_turn_end(state: State) -> bool:
    return state._playable_dice.sum() == -4


def _change_turn(state: State, key=None) -> State:
    board: Array = _flip_board(state._board)
    turn: Array = (state._turn + 1) % 2
    current_player: Array = (state.current_player + 1) % 2
    terminated: Array = state.terminated
    dice = jnp.zeros(2, dtype=jnp.int32)
    playable_dice = jnp.full(4, -1, dtype=jnp.int32)
    played_dice_num = jnp.int32(0)
    legal_action_mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
    return state.replace(
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
        _is_stochastic=jnp.array(True, dtype=jnp.bool_),
    )


def _roll_init_dice(rng: PRNGKey) -> Array:
    init_dice_pattern = jnp.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 3], [2, 4], [2, 5], [3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4]], dtype=jnp.int32)
    return jax.random.choice(rng, init_dice_pattern)


def _init_turn(dice: Array) -> Array:
    diff = dice[1] - dice[0]
    return jnp.int32(diff > 0)


def _set_playable_dice(dice: Array) -> Array:
    is_doubles = dice[0] == dice[1]
    doubles_result = jnp.full(4, dice[0], dtype=jnp.int32)
    non_doubles_result = jnp.array([dice[0], dice[1], -1, -1], dtype=jnp.int32)
    return jnp.where(is_doubles, doubles_result, non_doubles_result)


def _update_playable_dice(playable_dice: Array, played_dice_num: Array, dice: Array, action: Array) -> Array:
    played_die = action % 6
    is_doubles = dice[0] == dice[1]
    doubles_result = playable_dice.at[3 - played_dice_num].set(-1)
    matches = playable_dice == played_die
    first_match_idx = jnp.argmax(matches)
    non_doubles_result = playable_dice.at[first_match_idx].set(-1)
    return jnp.where(is_doubles, doubles_result, non_doubles_result)


def _home_board() -> Array:
    return jnp.arange(18, 24, dtype=jnp.int32)


def _off_idx() -> int:
    return 26


def _bar_idx() -> int:
    return 24


def _is_all_on_home_board(board: Array):
    home_board = _home_board()
    on_home_board = jnp.minimum(jnp.maximum(board[home_board], 0), 15).sum()
    off = board[_off_idx()]
    return (15 - off) == on_home_board


def _decompose_action(action: Array):
    return (
        _ACTION_SRC_LOOKUP[action],
        _ACTION_DIE_LOOKUP[action],
        _ACTION_TGT_LOOKUP[action]
    )


def _is_action_legal(board: Array, action: Array) -> bool:
    return _is_action_legal_branchless(board, action)


def _move(board: Array, action: Array) -> Array:
    src, _, tgt = _decompose_action(action)
    board = board.at[_bar_idx() + 1].add(-1 * (board[tgt] == -1))
    board = board.at[src].add(-1)
    board = board.at[tgt].add(1 + (board[tgt] == -1))
    return board


def _is_all_off(board: Array) -> bool:
    return board[_off_idx()] == 15


def _calc_win_score(board: Array) -> int:
    g = _is_gammon(board)
    return 1 + g + (g & _remains_at_inner(board))


def _is_gammon(board: Array) -> bool:
    return board[_off_idx() + 1] == 0


def _remains_at_inner(board: Array) -> bool:
    return jnp.take(board, _home_board()).sum() != 0


def _get_valid_sequence_mask(board: Array, die_first: int, die_second: int) -> Array:
    candidate_actions = _SRC_INDICES * 6 + (die_first - 1)
    is_legal_first_move = jax.vmap(_is_action_legal, in_axes=(None, 0))(board, candidate_actions)
    next_boards = jax.vmap(_move, in_axes=(None, 0))(board, candidate_actions)

    def _check_any_move(b, d):
        acts = _SRC_INDICES * 6 + (d - 1)
        return jax.vmap(_is_action_legal, in_axes=(None, 0))(b, acts).any()

    can_play_second_die = jax.vmap(_check_any_move, in_axes=(0, None))(next_boards, die_second)
    valid_candidates = is_legal_first_move & can_play_second_die
    full_mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
    return full_mask.at[candidate_actions].set(valid_candidates)


def _compute_two_dice_masks(board: Array, die1: int, die2: int):
    mask_d1_then_d2 = _get_valid_sequence_mask(board, die1, die2)
    mask_d2_then_d1 = _get_valid_sequence_mask(board, die2, die1)
    can_play_both = mask_d1_then_d2.any() | mask_d2_then_d1.any()
    return mask_d1_then_d2, mask_d2_then_d1, can_play_both


def _get_forced_single_move_mask(board: Array, die1: int, die2: int) -> Array:
    d_h = jnp.maximum(die1, die2)
    d_l = jnp.minimum(die1, die2)
    mask_h = _legal_action_mask_for_valid_single_dice(board, d_h - 1)
    can_play_h = mask_h.any()
    mask_l = _legal_action_mask_for_valid_single_dice(board, d_l - 1)
    return jax.lax.cond(can_play_h, lambda: mask_h, lambda: mask_l)


def _apply_special_backgammon_rules(board: Array, turn_dice: Array) -> Array:
    d1_0idx, d2_0idx = turn_dice[0], turn_dice[1]
    d1 = d1_0idx + 1
    d2 = d2_0idx + 1
    mask_d1_then_d2, mask_d2_then_d1, can_play_both = _compute_two_dice_masks(board, d1, d2)
    forced_full_move_mask = mask_d1_then_d2 | mask_d2_then_d1
    return jax.lax.cond(
        can_play_both,
        lambda: forced_full_move_mask,
        lambda: _get_forced_single_move_mask(board, d1, d2),
    )


def _legal_action_mask(board: Array, playable_dice: Array, turn_dice: Array, played_dice_num: Array) -> Array:
    dice_to_check = playable_dice[:2]
    unique_dice = jnp.unique(dice_to_check, size=2, fill_value=-1)
    unique_masks = jax.vmap(partial(_legal_action_mask_for_single_die_branchless, board=board))(die=unique_dice)
    simple_legal_mask = unique_masks.any(axis=0)
    is_start_of_turn = (played_dice_num == 0)
    is_nondoubles = (turn_dice[0] != turn_dice[1])

    def apply_rules_fn():
        return _apply_special_backgammon_rules(board, turn_dice)

    def keep_simple_mask_fn():
        return simple_legal_mask

    final_mask = jax.lax.cond(is_start_of_turn & is_nondoubles, apply_rules_fn, keep_simple_mask_fn)
    legal_action_exists = final_mask.any()
    return jax.lax.cond(legal_action_exists, lambda: final_mask, lambda: _NO_OP_MASK)


def _legal_action_mask_for_valid_single_dice(board: Array, die) -> Array:
    actions = _SRC_INDICES * 6 + die
    is_legal = jax.vmap(_is_action_legal, in_axes=(None, 0))(board, actions)
    mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
    return mask.at[actions].set(is_legal)


# Probability distributions
_STOCHASTIC_ACTION_PROBS = jnp.array([
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
    2/36, 2/36, 2/36, 2/36, 2/36,
    2/36, 2/36, 2/36, 2/36,
    2/36, 2/36, 2/36,
    2/36, 2/36,
    2/36,
], dtype=jnp.float32)

_STOCHASTIC_SIMPLE_DOUBLES_ACTION_PROBS = jnp.array([
    1/6, 1/6, 1/6, 1/6, 1/6, 1/6,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=jnp.float32)

_STOCHASTIC_DICE_MAPPING = jnp.array([
    [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 3], [2, 4], [2, 5],
    [3, 4], [3, 5],
    [4, 5],
], dtype=jnp.int32)
