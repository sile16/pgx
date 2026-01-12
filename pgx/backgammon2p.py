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
_POINT_INDICES = jnp.arange(24, dtype=jnp.int32)
_NO_OP_ACTION_MASK = jnp.zeros(26 * 26, dtype=jnp.bool_).at[0].set(True)

# Precompute lookup tables
_ACTIONS = jnp.arange(26 * 26, dtype=jnp.int32)
_ACTION_SRC1 = _ACTIONS // 26
_ACTION_SRC2 = _ACTIONS % 26

_HOME_BOARD_INDICES = jnp.arange(18, 24, dtype=jnp.int32)

def _build_tgt_map():
    # src_idx: 0..25, die: 1..6
    # map to tgt (0..27, where 26=off, 27=invalid?)
    # reusing _src_from_idx and _tgt_from_src_die logic but precomputing
    src_indices = jnp.arange(26, dtype=jnp.int32)
    die_values = jnp.arange(1, 7, dtype=jnp.int32)
    
    def get_tgt(src_idx, die):
        src = jnp.where(src_idx == 0, -2, jnp.where(src_idx == 1, 24, src_idx - 2))
        tgt_from_bar = die - 1
        tgt_from_board = src + die
        tgt_normal = jnp.where((tgt_from_board >= 0) & (tgt_from_board <= 23), tgt_from_board, 26)
        tgt = jnp.where(src >= 24, tgt_from_bar, tgt_normal)
        tgt = jnp.where(src == -2, -2, tgt)
        return tgt

    # Shape: (26, 7) - index 0 is dummy die=0
    return jax.vmap(lambda s: jax.vmap(lambda d: get_tgt(s, d))(die_values))(src_indices)

_TGT_MAP = _build_tgt_map() # (26, 6) corresponding to die 1..6

# Precomputed interaction tables for legal action mask optimization
def _build_interaction_tables():
    """Build tables for move interactions at module load time."""
    # _TGT_AS_SRC_IDX[s, die_idx] -> src_idx that corresponds to tgt(s, die), or -1 if off/invalid
    # If tgt is a point (0-23), src_idx = tgt + 2
    # If tgt is off (26) or invalid (-2), src_idx = -1
    def tgt_to_src_idx(tgt):
        return jnp.where((tgt >= 0) & (tgt <= 23), tgt + 2, -1)

    tgt_as_src = jax.vmap(lambda s: jax.vmap(tgt_to_src_idx)(_TGT_MAP[s]))(_SRC_INDICES)  # (26, 6)

    # _CHAIN_MASK[s1, die_idx, s2] -> True if tgt(s1, die) corresponds to src(s2)
    # This means moving s1 with die lands on the point that s2 would move from
    chain_mask = jnp.zeros((26, 6, 26), dtype=bool)
    for die_idx in range(6):
        # For each s1, check if tgt_as_src[s1, die_idx] == s2
        chain_mask = chain_mask.at[:, die_idx, :].set(
            tgt_as_src[:, die_idx:die_idx+1] == _SRC_INDICES[None, :]
        )

    return tgt_as_src, chain_mask

_TGT_AS_SRC_IDX, _CHAIN_MASK = _build_interaction_tables()

# Same-source mask: diagonal where s1 == s2 (excluding pass)
_SAME_SOURCE_MASK = (jnp.arange(26)[:, None] == jnp.arange(26)[None, :]) & (jnp.arange(26)[:, None] > 0)

# Bar index in src_idx encoding
_BAR_SRC_IDX = 1


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
        if key is None:
            raise ValueError("Backgammon2P.step requires a PRNG key for stochastic dice rolls.")

        no_moves = (~_has_playable_action(state.legal_action_mask)) & (~state._is_stochastic)
        base_step = super(Backgammon2P, self).step

        def auto_skip_turn():
            skipped_state = _auto_skip_no_moves(self, state.replace(_step_count=state._step_count + 1), key)
            skipped_state = jax.lax.cond(
                skipped_state.terminated,
                lambda: skipped_state.replace(legal_action_mask=jnp.ones_like(skipped_state.legal_action_mask)),
                lambda: skipped_state,
            )
            observation = self.observe(skipped_state)
            return skipped_state.replace(observation=observation)

        return jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: base_step(state, action, key),
            lambda: jax.lax.cond(no_moves, auto_skip_turn, lambda: base_step(state, action, key)),
        )

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

    def _has_playable_actions(self, state: State) -> Array:
        return _has_playable_action(state.legal_action_mask)

    def _auto_advance_no_playable(self, state: State, key: PRNGKey) -> State:
        # Auto-pass and flip the turn when only pass actions are available.
        state = state.replace(_step_count=state._step_count + 1)
        return _change_turn(state)


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
    board_after_move = _apply_action_branchless(state._board, src1_idx, src2_idx, die1, die2)
    remaining_actions_next = state._remaining_actions - 1
    legal_action_mask_next = _legal_action_mask(board_after_move, state._dice)
    no_moves_left = ~legal_action_mask_next.any()
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
    inner_sum = jnp.take(board, _home_board()).sum()
    on_bar = board[_bar_idx() + 1] != 0
    return (inner_sum != 0) | on_bar


def _home_board() -> Array:
    return jnp.arange(18, 24, dtype=jnp.int32)


def _off_idx() -> int:
    return 26


def _bar_idx() -> int:
    return 24


def _decode_action(action: Array):
    return _ACTION_SRC1[action], _ACTION_SRC2[action]


def _src_from_idx(src_idx: Array) -> Array:
    # Used for logic where we need actual board index
    return jnp.where(src_idx == 0, -2, jnp.where(src_idx == 1, 24, src_idx - 2))


def _tgt_from_idx_die(src_idx: Array, die: Array) -> Array:
    # Look up target from precomputed table
    # die is 1-based, table is 0-based so die-1
    return _TGT_MAP[src_idx, die - 1]

def _board_invariants(board: Array) -> Tuple[Array, Array, Array]:
    """Features that are constant for all move checks on a given board."""
    bar_clear = board[_bar_idx()] == 0
    on_home_board = jnp.where(board[_HOME_BOARD_INDICES] > 0, board[_HOME_BOARD_INDICES], 0).sum()
    off_count = board[_off_idx()]
    all_on_home = (15 - off_count) == on_home_board
    has_checker = board[:24] > 0
    rear_pos = jnp.where(has_checker, _POINT_INDICES, jnp.int32(100))
    rear_idx = jnp.min(rear_pos)
    rear_dist = 24 - rear_idx
    return bar_clear, all_on_home, rear_dist


def _is_move_legal_branchless(
    board: Array,
    src_idx: Array,
    die: Array,
    board_invariants: Optional[Tuple[Array, Array, Array]] = None,
) -> Array:
    if board_invariants is None:
        board_invariants = _board_invariants(board)
    bar_clear, all_on_home, rear_dist = board_invariants
    src = _src_from_idx(src_idx)
    tgt = _tgt_from_idx_die(src_idx, die)

    src_safe = jnp.clip(src, 0, 27)
    tgt_safe = jnp.clip(tgt, 0, 27)

    is_to_point = (tgt >= 0) & (tgt <= 23) & (src >= 0)
    is_to_off = (tgt == 26) & (src >= 0)

    src_has_checker = board[src_safe] >= 1
    tgt_is_open = board[tgt_safe] >= -1
    from_bar = src >= 24

    point_legal = src_has_checker & tgt_is_open & (from_bar | bar_clear)

    dist = 24 - src

    exact_roll = dist == die
    highest_and_overbear = (rear_dist <= die) & (rear_dist == dist)
    off_legal = src_has_checker & all_on_home & (exact_roll | highest_and_overbear)

    is_no_op = src == -2
    # Result is true if (to_point & point_legal) OR (to_off & off_legal)
    result = (is_to_point & point_legal) | (is_to_off & off_legal)
    return result & (~is_no_op)


def _apply_move_branchless(board: Array, src_idx: Array, die: Array) -> Array:
    src = _src_from_idx(src_idx)
    tgt = _tgt_from_idx_die(src_idx, die)

    # Logic:
    # if src_idx != 0:
    #   bar_adj = -1 if tgt == -1 (hit) else 0
    #   src_adj = -1
    #   tgt_adj = 1 + (1 if tgt == -1 else 0)
    # else:
    #   all 0
    
    is_move = src_idx != 0
    is_hit = board[tgt] == -1
    
    bar_adj = (is_move & is_hit) * -1 # Remove opponent from their point (add -1 to bar? No. Wait.)
    # Board representation: 
    # Positive = My checkers. Negative = Opponent checkers.
    # Bar = Index 24 (Mine), 25 (Opponent).
    # Hit: Move opponent checker to their bar (Index 25).
    # Since board is flipped for me, opponent bar is 25.
    # Opponent checker is -1. moving it to bar means:
    # board[tgt] goes from -1 to 0 (removed).
    # board[25] goes from X to X-1 (adds 1 opponent checker).
    # _apply_move original:
    # b = b.at[_bar_idx() + 1].add(-1 * (b[tgt] == -1))
    # b = b.at[src].add(-1)
    # b = b.at[tgt].add(1 + (b[tgt] == -1))
    
    # Branchless updates:
    # We can use .at[...].add(...) even with invalid indices if we mask the values 
    # or if we rely on src/tgt being valid when is_move is true.
    # src, tgt are safe? 
    # If src_idx=0, src=-2. board[-2] is valid (wraps) but we shouldn't modify.
    # So we must use jnp.where or multiply delta by is_move.
    
    # Let's compute deltas for specific indices.
    # Since scatter-add is expensive, maybe just updating the 3 positions is better?
    # But indices are dynamic.
    
    # We'll use a sequence of updates where the delta is 0 if is_move is false.
    
    # 1. Update Opponent Bar (25)
    # delta = -1 if (is_move and is_hit) else 0
    delta_opp_bar = jnp.where(is_move & is_hit, -1, 0).astype(jnp.int32)
    board = board.at[25].add(delta_opp_bar)
    
    # 2. Update Source
    # delta = -1 if is_move else 0
    # Use safe index if not move
    safe_src = jnp.where(is_move, src, 0)
    delta_src = jnp.where(is_move, -1, 0).astype(jnp.int32)
    board = board.at[safe_src].add(delta_src)
    
    # 3. Update Target
    # delta = 1 + (1 if is_hit else 0) -> 1 or 2
    # if is_hit, board[tgt] was -1. adding 2 makes it 1.
    # if not hit, board[tgt] was N. adding 1 makes it N+1.
    safe_tgt = jnp.where(is_move, tgt, 0)
    delta_tgt = jnp.where(is_move, 1 + is_hit.astype(jnp.int32), 0).astype(jnp.int32)
    board = board.at[safe_tgt].add(delta_tgt)
    
    return board


def _apply_action_branchless(board: Array, src1_idx: Array, src2_idx: Array, die1: Array, die2: Array) -> Array:
    # Execute both moves; choose reverse order when only that sequence is legal.
    board_seq1_first = _apply_move_branchless(board, src1_idx, die1)
    board_seq1 = _apply_move_branchless(board_seq1_first, src2_idx, die2)

    board_seq2_first = _apply_move_branchless(board, src2_idx, die2)
    board_seq2 = _apply_move_branchless(board_seq2_first, src1_idx, die1)

    base_inv = _board_invariants(board)
    seq1_first_legal = _is_move_legal_branchless(board, src1_idx, die1, base_inv)
    seq2_first_legal = _is_move_legal_branchless(board, src2_idx, die2, base_inv)

    seq1_second_legal = _is_move_legal_branchless(
        board_seq1_first, src2_idx, die2, _board_invariants(board_seq1_first)
    )
    seq2_second_legal = _is_move_legal_branchless(
        board_seq2_first, src1_idx, die1, _board_invariants(board_seq2_first)
    )

    seq1_legal = seq1_first_legal & seq1_second_legal
    seq2_legal = seq2_first_legal & seq2_second_legal

    use_seq1 = seq1_legal | (~seq2_legal)  # Prefer seq1 when both legal; fall back to seq2 when only reverse works.
    return jnp.where(use_seq1[..., None], board_seq1, board_seq2)


def _legal_action_mask_nondoubles(board: Array, die1: Array, die2: Array) -> Array:
    """
    Compute legal action mask using precomputed interaction tables.

    Uses outer product + corrections instead of board modifications.
    23% faster at batch size 4000 due to reduced memory bandwidth.
    """
    base_inv = _board_invariants(board)
    bar_clear, all_on_home, rear_dist = base_inv

    # Step 1: Compute independent single-move legality for each die
    def check_both_dice(s):
        l1 = _is_move_legal_branchless(board, s, die1, base_inv)
        l2 = _is_move_legal_branchless(board, s, die2, base_inv)
        return l1, l2

    legal_both = jax.vmap(check_both_dice)(_SRC_INDICES)
    legal_d1 = legal_both[0]  # (26,)
    legal_d2 = legal_both[1]  # (26,)

    # Step 2: Get checker counts and target openness for corrections
    def get_src_info(src_idx):
        src = _src_from_idx(src_idx)
        src_safe = jnp.clip(src, 0, 27)
        count = jnp.where(src >= 0, board[src_safe], 0)
        return count

    checker_counts = jax.vmap(get_src_info)(_SRC_INDICES)  # (26,)

    # Check if targets are open (for chain move validation)
    def check_tgt_open(src_idx, die):
        tgt = _tgt_from_idx_die(src_idx, die)
        tgt_safe = jnp.clip(tgt, 0, 27)
        is_to_point = (tgt >= 0) & (tgt <= 23)
        is_to_off = tgt == 26
        tgt_open = board[tgt_safe] >= -1
        src = _src_from_idx(src_idx)
        dist = 24 - src
        exact_roll = dist == die
        can_overbear = (rear_dist <= die) & (rear_dist == dist)
        off_ok = all_on_home & (exact_roll | can_overbear)
        return (is_to_point & tgt_open) | (is_to_off & off_ok)

    tgt_open_d1 = jax.vmap(lambda s: check_tgt_open(s, die1))(_SRC_INDICES)
    tgt_open_d2 = jax.vmap(lambda s: check_tgt_open(s, die2))(_SRC_INDICES)

    # Step 3: Outer product - base joint legality
    joint_fwd = legal_d1[:, None] & legal_d2[None, :]  # (26, 26) [s1, s2]
    joint_rev = legal_d2[:, None] & legal_d1[None, :]  # (26, 26) [s2, s1]

    # Step 4: Apply corrections

    # Correction 1: Same source needs 2+ checkers
    same_src_needs_two = _SAME_SOURCE_MASK & (checker_counts[:, None] < 2)
    joint_fwd = joint_fwd & ~same_src_needs_two
    joint_rev = joint_rev & ~same_src_needs_two

    # Correction 2: Chain moves - target of first becomes source of second
    die1_idx = die1 - 1
    die2_idx = die2 - 1
    chain_fwd = _CHAIN_MASK[:, die1_idx, :]
    chain_rev = _CHAIN_MASK[:, die2_idx, :]

    src_has_zero = checker_counts == 0
    s1_is_bar = _SRC_INDICES == _BAR_SRC_IDX
    s2_is_bar = _SRC_INDICES == _BAR_SRC_IDX
    bar_has_one = board[24] == 1

    bar_clear_after_s1 = bar_clear | (s1_is_bar & bar_has_one)
    s2_bar_ok = s2_is_bar | bar_clear_after_s1[:, None]

    chain_enable_fwd = (
        legal_d1[:, None] & chain_fwd & src_has_zero[None, :] &
        tgt_open_d2[None, :] & s2_bar_ok
    )
    joint_fwd = joint_fwd | chain_enable_fwd

    bar_clear_after_s2 = bar_clear | (s2_is_bar & bar_has_one)
    s1_bar_ok = s1_is_bar[None, :] | bar_clear_after_s2[:, None]

    chain_enable_rev = (
        legal_d2[:, None] & chain_rev & src_has_zero[None, :] &
        tgt_open_d1[None, :] & s1_bar_ok
    )
    joint_rev = joint_rev | chain_enable_rev

    # Correction 3: Bar clearing enables non-bar moves
    inv_bar_clear = (True, all_on_home, rear_dist)
    legal_d2_if_bar_clear = jax.vmap(
        lambda s: _is_move_legal_branchless(board, s, die2, inv_bar_clear)
    )(_SRC_INDICES)
    newly_legal_d2 = legal_d2_if_bar_clear & ~legal_d2

    bar_clear_enables_fwd = (
        s1_is_bar[:, None] & bar_has_one & legal_d1[:, None] & newly_legal_d2[None, :]
    )
    joint_fwd = joint_fwd | bar_clear_enables_fwd

    legal_d1_if_bar_clear = jax.vmap(
        lambda s: _is_move_legal_branchless(board, s, die1, inv_bar_clear)
    )(_SRC_INDICES)
    newly_legal_d1 = legal_d1_if_bar_clear & ~legal_d1

    bar_clear_enables_rev = (
        s2_is_bar[:, None] & bar_has_one & legal_d2[:, None] & newly_legal_d1[None, :]
    )
    joint_rev = joint_rev | bar_clear_enables_rev

    # Step 5: Combine forward and reverse
    joint_rev_transposed = joint_rev.T
    can_play_both_mask = (joint_fwd | joint_rev_transposed).reshape(26 * 26)
    can_play_both = can_play_both_mask.any()

    # Step 6: Single move logic
    can_play_d1 = legal_d1.any()
    can_play_d2 = legal_d2.any()

    s2_is_pass = _ACTION_SRC2 == 0
    s1_is_pass = _ACTION_SRC1 == 0

    legal1_expanded = jnp.repeat(legal_d1, 26)
    legal2_expanded = jnp.tile(legal_d2, 26)

    single_d1_mask = legal1_expanded & s2_is_pass
    single_d2_mask = legal2_expanded & s1_is_pass

    single_mask = jnp.where(can_play_d2, single_d2_mask, single_d1_mask)
    final_mask = jnp.where(can_play_both, can_play_both_mask, single_mask)

    no_moves = ~final_mask.any()
    pass_action = s1_is_pass & s2_is_pass

    return jnp.where(no_moves, pass_action, final_mask)


def _legal_action_mask_doubles(board: Array, die: Array) -> Array:
    # Die1 = Die2 = Die.
    # Logic is similar but simpler: no reverse order check needed (identical dice).
    # But wait, standard backgammon doubles allow 4 moves.
    # This env encodes doubles as 2 steps of 2 moves.
    # So "One Step" = 2 moves.
    
    # We need to find if we can play 2 moves: s1, s2.
    # Or if we can only play 1 move: s1 (s2=Pass) or s2 (s1=Pass)?
    # Since dice are identical, order doesn't matter for (s1, Pass) vs (Pass, s1).
    # But for (s1, s2), we must check s1->s2.
    
    # 1. Legal s1
    base_invariants = _board_invariants(board)
    legal_srcs = jax.vmap(
        lambda s: _is_move_legal_branchless(board, s, die, base_invariants)
    )(_SRC_INDICES)
    can_play_one = legal_srcs.any()
    
    # 2. Legal sequence s1->s2
    next_boards = jax.vmap(lambda s: _apply_move_branchless(board, s, die))(_SRC_INDICES)
    next_board_invariants = jax.vmap(_board_invariants)(next_boards)
    legal_seq_matrix = jax.vmap(
        lambda nb, invariants: jax.vmap(
            lambda s: _is_move_legal_branchless(nb, s, die, invariants)
        )(_SRC_INDICES)
    )(next_boards, next_board_invariants)
    
    legal_seq = legal_seq_matrix.reshape(26 * 26)
    legal1_expanded = jnp.repeat(legal_srcs, 26)
    
    two_moves_legal = legal1_expanded & legal_seq
    
    # Permutation symmetry for doubles?
    # Action (A, B) and (B, A) are distinct in encoding.
    # If A != B, and we can play A then B:
    # (A, B) is legal.
    # Is (B, A) legal? (Play B then A).
    # Yes, usually unless A unblocks B or B unblocks A.
    # The action encoding represents "Source for Die 1, Source for Die 2".
    # Since Die 1 == Die 2, (A, B) means "Move A then Move B" OR "Move B then Move A"?
    # In `_apply_action`, if is_doubles, we try both sequences?
    # Original code:
    # seq1_legal = ...
    # return cond(seq1_legal, seq1, seq2)
    # So it tries to execute as-ordered. If (A, B) is requested, it tries A->B.
    # So we must ensure A->B is legal for action (A, B) to be valid.
    # (B, A) would be B->A.
    # So `two_moves_legal` covers it.
    
    can_play_two = two_moves_legal.any()
    
    # Single Move Logic
    # If we can play 2, we must.
    # If we can't play 2, but can play 1:
    # Allow (A, Pass) or (Pass, A)?
    # Convention: usually canonicalize or allow both.
    # Original code: ((src1 == 0) & move2) | ((src2 == 0) & move1)
    
    s2_is_pass = _ACTION_SRC2 == 0
    s1_is_pass = _ACTION_SRC1 == 0
    
    # Check legal1_expanded for s1 (move1)
    # Check legal2_expanded for s2 (move2)
    legal2_expanded = jnp.tile(legal_srcs, 26)
    
    single_mask = (legal1_expanded & s2_is_pass) | (legal2_expanded & s1_is_pass)
    
    final_mask = jnp.where(can_play_two, two_moves_legal, jnp.where(can_play_one, single_mask, FALSE))
    
    # No moves
    no_moves = ~final_mask.any()
    pass_action = (s1_is_pass & s2_is_pass)
    
    return jnp.where(no_moves, pass_action, final_mask)


def _legal_action_mask(board: Array, dice: Array) -> Array:
    is_doubles = dice[0] == dice[1]
    die1 = dice[0] + 1
    die2 = dice[1] + 1
    return jax.lax.cond(
        is_doubles,
        lambda: _legal_action_mask_doubles(board, die1),
        lambda: _legal_action_mask_nondoubles(board, die1, die2),
    )


def _has_playable_action(mask: Array) -> Array:
    # Ignore the pass slot when checking for available moves.
    non_pass_mask = mask & (~_NO_OP_ACTION_MASK)
    return non_pass_mask.any()


def _auto_skip_no_moves(env: Backgammon2P, state: State, key: Array) -> State:
    """Automatically advance turns when no playable actions exist."""

    def cond(carry):
        state, key = carry
        still_playing = ~(state.terminated | state.truncated)
        no_moves = ~_has_playable_action(state.legal_action_mask)
        return still_playing & no_moves

    def body(carry):
        state, key = carry
        key, roll_key = jax.random.split(key)
        state = state.replace(_step_count=state._step_count + 1)
        state = _change_turn(state)
        state = env._step_stochastic_random(state, roll_key)
        return state, key

    state, _ = jax.lax.while_loop(cond, body, (state, key))
    return state


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
