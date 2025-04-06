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

from functools import partial
from typing import Optional
import dataclasses

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
    observation: Array = jnp.zeros(34, dtype=jnp.int32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    is_stochastic: Array = jnp.array(True, dtype=jnp.bool_)  # whether the state is in stochastic mode
    # micro action = 6 * src + die
    legal_action_mask: Array = jnp.zeros(6 * 26, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Backgammon specific ---
    # points(24) bar(2) off(2). black+, white-
    _board: Array = jnp.zeros(28, dtype=jnp.int32)
    _dice: Array = jnp.zeros(2, dtype=jnp.int32)  # 0~5: 1~6
    _playable_dice: Array = jnp.zeros(4, dtype=jnp.int32)  # playable dice -1 for empty
    _played_dice_num: Array = jnp.int32(0)  # the number of dice played
    _turn: Array = jnp.int32(1)  # black: 0 white:1
    

    @property
    def env_id(self) -> core.EnvId:
        return "backgammon"

    def replace(self, **kwargs) -> "State":
        """Create a new state with updated fields."""
        return dataclasses.replace(self, **kwargs)


class Backgammon(core.Env):
    def __init__(self, simple_doubles: bool = False):
        super().__init__()
        self.simple_doubles = simple_doubles
        self.stochastic_action_probs = (
            _STOCHASTIC_SIMPLE_DOUBLES_ACTION_PROBS if simple_doubles 
            else _STOCHASTIC_ACTION_PROBS
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

    def _observe(self, state: core.State, player_id: Optional[Array] = None) -> Array:
        """
        Return observation for current player
        
        The player_id parameter is deprecated and will be removed in the future.
        The method ignores player_id and always returns the observation for the current player.
        """
        assert isinstance(state, State)
        # We're ignoring player_id since the standalone _observe function doesn't use it
        # and the base class observe() method already issues the deprecation warning
        return _observe(state)
    
    def set_dice(self, state: State, dice: Array) -> State:
        """
        Use for setting the dice for testing or using external dice.
        dice is a 2 digit array 0-5, 0 for 1, 1 for 2, etc.
        """
        playable_dice: Array = _set_playable_dice(dice)
        played_dice_num: Array = jnp.int32(0)
        legal_action_mask: Array = _legal_action_mask(state._board, dice)
        return state.replace(  # type: ignore
            current_player=state.current_player,
            _board=state._board,
            terminated=state.terminated,
            _turn=state._turn,
            _dice=dice,
            _playable_dice=playable_dice,
            _played_dice_num=played_dice_num,
            legal_action_mask=legal_action_mask,
            is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )
    
    def stochastic_step(self, state: State, action: Array) -> State:
        """
        Handle a stochastic step (dice roll) for programs that want to control dice rolls.
        This is separate from the regular step function to maintain backward compatibility.
        
        Args:
            state: Current game state
            action: Index into _STOCHASTIC_DICE_MAPPING (0-20) representing the dice roll
            
        Returns:
            New state with dice set and is_stochastic set to False
        """
        # Get the dice roll from the mapping
        roll = _STOCHASTIC_DICE_MAPPING[action]
        return self.set_dice(state, roll)

    @property
    def id(self) -> core.EnvId:
        return "backgammon"

    @property
    def version(self) -> str:
        return "v2"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0


def _init(rng: PRNGKey) -> State:
    rng1, rng2 = jax.random.split(rng, num=2)
    current_player: Array = jax.random.bernoulli(rng1).astype(jnp.int32)
    board: Array = _make_init_board()
    terminated: Array = FALSE
    dice: Array = _roll_init_dice(rng2)
    playable_dice: Array = _set_playable_dice(dice)
    played_dice_num: Array = jnp.int32(0)
    turn: Array = _init_turn(dice)
    legal_action_mask: Array = _legal_action_mask(board, playable_dice)
    state = State(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        _turn=turn,
        legal_action_mask=legal_action_mask,
        is_stochastic=jnp.array(True, dtype=jnp.bool_)  #initial state is stochastic, as it requires a dice roll
    )
    return state


def _step(state: State, action: Array, key) -> State:
    """
    Step when not terminated
    """
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state._board),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state, action, key),
    )


def _observe(state: State) -> Array:
    """
    Return observation for current player
    """
    board: Array = state._board
    playable_dice_count_vec: Array = _to_playable_dice_count(
        state._playable_dice
    )  # 6 dim vec which represents the count of playable die.
    
    return jnp.concatenate((board, playable_dice_count_vec), axis=None)


def _to_playable_dice_count(playable_dice: Array) -> Array:
    """
    Return 6 dim vec which represents the number of playable die
    Examples
    Playable dice: 2, 3
    Return: [0, 1, 1, 0, 0, 0]

    Playable dice: 4, 4, 4, 4
    Return: [0, 0, 0, 0, 4, 0]
    """
    dice_indices: Array = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # maximum number of playable dice is 4

    def _insert_dice_num(idx: Array, playable_dice: Array) -> Array:
        vec: Array = jnp.zeros(6, dtype=jnp.int32)
        return (playable_dice[idx] != -1) * vec.at[playable_dice[idx]].set(1) + (playable_dice[idx] == -1) * vec

    return jax.vmap(_insert_dice_num)(dice_indices, jnp.tile(playable_dice, (4, 1))).sum(axis=0, dtype=jnp.int32)


def _winning_step(
    state: State,
) -> State:
    """
    Step with winner
    """
    win_score = _calc_win_score(state._board)
    winner = state.current_player
    loser = 1 - winner
    reward = jnp.ones_like(state.rewards)
    reward = reward.at[winner].set(win_score)
    reward = reward.at[loser].set(-win_score)
    state = state.replace(terminated=TRUE)  # type: ignore
    return state.replace(rewards=reward)  # type: ignore


def _no_winning_step(state: State, action: Array, key) -> State:
    """
    Step with no winner. Change turn if turn end condition is satisfied.
    """
    return jax.lax.cond(
        (_is_turn_end(state) | (action // 6 == 0)),
        lambda: _change_turn(state, key),
        lambda: state,
    )


def _update_by_action(state: State, action: Array) -> State:
    """
    Update state by action
    """
    is_no_op = action // 6 == 0
    current_player: Array = state.current_player
    terminated: Array = state.terminated
    board: Array = _move(state._board, action)
    played_dice_num: Array = jnp.int32(state._played_dice_num + 1)
    playable_dice: Array = _update_playable_dice(state._playable_dice, state._played_dice_num, state._dice, action)
    legal_action_mask: Array = _legal_action_mask(board, playable_dice)
    return jax.lax.cond(
        is_no_op,
        lambda: state,
        lambda: state.replace(  # type: ignore
            current_player=current_player,
            terminated=terminated,
            _board=board,
            _turn=state._turn,
            _dice=state._dice,
            _playable_dice=playable_dice,
            _played_dice_num=played_dice_num,
            legal_action_mask=legal_action_mask,
        ),
    )  # no-opの時はupdateしない


def _flip_board(board):
    """
    Flip a board when turn changes. Multiply -1 to the board so that we can always consider the board from black's perspective.
    """
    _board = board
    board = board.at[:24].set(jnp.flip(_board[:24]))
    board = board.at[24:26].set(jnp.flip(_board[24:26]))
    board = board.at[26:28].set(jnp.flip(_board[26:28]))
    return -1 * board


def _make_init_board() -> Array:
    """
    Initialize the board based on black's perspective.
    """
    board: Array = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int32)  # type: ignore
    return board


def _is_turn_end(state: State) -> bool:
    """
    Turn will end if there is no playable dice or no legal action.
    """
    return state._playable_dice.sum() == -4  # type: ignore


def _change_turn(state: State, key) -> State:
    """
    Change turn and return new state.
    """
    board: Array = _flip_board(state._board)
    turn: Array = (state._turn + 1) % 2
    current_player: Array = (state.current_player + 1) % 2
    terminated: Array = state.terminated
    dice: Array = _roll_dice(key)
    playable_dice: Array = _set_playable_dice(dice)
    played_dice_num: Array = jnp.int32(0)
    legal_action_mask: Array = _legal_action_mask(board, dice)
    return state.replace(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
        is_stochastic=jnp.array(True, dtype=jnp.bool_) , #after a player change it's a stochastic state
    )


def _roll_init_dice(rng: PRNGKey) -> Array:
    """
    Roll till the dice are different.
    """

    init_dice_pattern: Array = jnp.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 3], [2, 4], [2, 5], [3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4]], dtype=jnp.int32)  # type: ignore
    return jax.random.choice(rng, init_dice_pattern)


def _roll_dice(rng: PRNGKey) -> Array:
    roll: Array = jax.random.randint(rng, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int32)
    return roll[0]


def _init_turn(dice: Array) -> Array:
    """
    Decide turn at the beginning of the game.
    Begin with those who have bigger dice
    """
    diff = dice[1] - dice[0]
    return jnp.int32(diff > 0)


def _set_playable_dice(dice: Array) -> Array:
    """
    -1 for empty
    """
    return (dice[0] == dice[1]) * jnp.array([dice[0]] * 4, dtype=jnp.int32) + (dice[0] != dice[1]) * jnp.array(
        [dice[0], dice[1], -1, -1], dtype=jnp.int32
    )


def _update_playable_dice(
    playable_dice: Array,
    played_dice_num: Array,
    dice: Array,
    action: Array,
) -> Array:
    _n = played_dice_num
    die_array = jnp.array([action % 6] * 4, dtype=jnp.int32)
    dice_indices: Array = jnp.array([0, 1, 2, 3], dtype=jnp.int32)  # maximum number of playable dice is 4

    def _update_for_diff_dice(die: Array, idx: Array, playable_dice: Array):
        return (die == playable_dice[idx]) * -1 + (die != playable_dice[idx]) * playable_dice[idx]

    return (dice[0] == dice[1]) * playable_dice.at[3 - _n].set(-1) + (dice[0] != dice[1]) * jax.vmap(
        _update_for_diff_dice
    )(die_array, dice_indices, jnp.tile(playable_dice, (4, 1))).astype(jnp.int32)


def _home_board() -> Array:
    """
    black: [18~23], white: [0~5]: Always black's perspective
    """
    return jnp.arange(18, 24, dtype=jnp.int32)  # type: ignore


def _off_idx() -> int:
    """
    black: 26, white: 27: Always black's perspective
    """
    return 26  # type: ignore


def _bar_idx() -> int:
    """
    black: 24, white 25: Always black's perspective
    """
    return 24  # type: ignore


def _rear_distance(board: Array) -> Array:
    """
    The distance from the farthest checker to the goal: Always black's perspective
    """
    b = board[:24]
    exists = jnp.where((b > 0), size=24, fill_value=jnp.nan)[0]  # type: ignore
    return 24 - jnp.min(jnp.nan_to_num(exists, nan=jnp.int32(100)))


def _is_all_on_home_board(board: Array):
    """
    One can bear off if all checkers are on home board.
    """
    home_board: Array = _home_board()
    on_home_board = jnp.minimum(jnp.maximum(board[home_board], 0), 15).sum()
    off = board[_off_idx()]  # type: ignore
    return (15 - off) == on_home_board


def _is_open(board: Array, point: int) -> bool:
    """
    Check if the point is open for the current player: Always black's perspective
    One can move to the point if there is no more than one opponent's checker.
    """
    checkers = board[point]
    return checkers >= -1  # type: ignore


def _exists(board: Array, point: int) -> bool:
    """
    Check if the point has the current player's checker: Always black's perspective
    """
    checkers = board[point]
    return checkers >= 1  # type: ignore


def _calc_src(src: Array) -> int:
    """
    Translate src to board index.
    """
    return (src == 1) * jnp.int32(_bar_idx()) + (src != 1) * jnp.int32(src - 2)  # type: ignore


def _calc_tgt(src: int, die) -> int:
    """
    Translate tgt to board index.
    """
    return (src >= 24) * (jnp.int32(die) - 1) + (src < 24) * jnp.int32(_from_board(src, die))  # type: ignore


def _from_board(src: int, die: int) -> int:
    _is_to_board = (src + die >= 0) & (src + die <= 23)
    return _is_to_board * jnp.int32(src + die) + ((~_is_to_board)) * jnp.int32(_off_idx())  # type: ignore


def _decompose_action(action: Array):
    """
    Decompose action to src, die, tgt.
    action = src*6 + die
    """
    src = _calc_src(action // 6)  # 0~25
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, die)
    return src, die, tgt


def _is_action_legal(board: Array, action: Array) -> bool:
    """
    Check if the action is legal.
    action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt = _decompose_action(action)
    _is_to_point = (0 <= tgt) & (tgt <= 23) & (src >= 0)
    return _is_to_point & _is_to_point_legal(board, src, tgt) | (~_is_to_point) & _is_to_off_legal(
        board, src, tgt, die
    )  # type: ignore


def _distance_to_goal(src: int) -> int:
    """
    The distance from the src to the goal: Always black's perspective
    """
    return 24 - src  # type: ignore


def _is_to_off_legal(board: Array, src: int, tgt: int, die: int):
    """
    Check if the action is legal when the target is off.
    The conditions are:
    1. src has checkers.
    2. All checkers are on home board.
    3. The distance from the src to the goal is the same as the die or the src is the farthest checker and the die is bigger than the distance.
    """
    r = _rear_distance(board)
    d = _distance_to_goal(src)
    return (
        (src >= 0) & _exists(board, src) & _is_all_on_home_board(board) & ((d == die) | ((r <= die) & (r == d)))
    )  # type: ignore


def _is_to_point_legal(board: Array, src: int, tgt: int) -> bool:
    """
    Check if the action is legal when the target is point.
    """
    e = _exists(board, src)
    o = _is_open(board, tgt)
    return ((src >= 24) & e & o) | ((src < 24) & e & o & (board[_bar_idx()] == 0))  # type: ignore


def _move(board: Array, action: Array) -> Array:
    """
    Move checkers based on the action.
    """
    src, _, tgt = _decompose_action(action)
    board = board.at[_bar_idx() + 1].add(
        -1 * (board[tgt] == -1)
    )  # If there is an opponent's checker on the target, hit it
    board = board.at[src].add(-1)
    board = board.at[tgt].add(1 + (board[tgt] == -1))  # If hit, the sign changes, so add 1
    return board


def _is_all_off(board: Array) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる. 常に黒視点
    If all checkers are off, the player wins. Always black's perspective.
    """
    return board[_off_idx()] == 15  # type: ignore


def _calc_win_score(board: Array) -> int:
    """
    Normal win: 1 point
    Gammon win: 2 points
    Backgammon win: 3 points
    """
    g = _is_gammon(board)
    return 1 + g + (g & _remains_at_inner(board))


def _is_gammon(board: Array) -> bool:
    """
    If there is no opponent's checker on off, the player wins gammon.
    """
    return board[_off_idx() + 1] == 0  # type: ignore


def _remains_at_inner(board: Array) -> bool:
    """
    (1) If there is no opponent's checker on off and (2) there is at least one opponent's checker on inner, the player wins backgammon.
    """
    return jnp.take(board, _home_board()).sum() != 0  # type: ignore


def _legal_action_mask(board: Array, dice: Array) -> Array:
    no_op_mask = jnp.zeros(26 * 6, dtype=jnp.bool_).at[0:6].set(TRUE)
    legal_action_mask = jax.vmap(partial(_legal_action_mask_for_single_die, board=board))(die=dice).any(
        axis=0
    )  # (26 * 6)
    legal_action_exists = ~(legal_action_mask.sum() == 0)
    return (
        legal_action_exists * legal_action_mask + ~legal_action_exists * no_op_mask
    )  # if there is no legal action, no-op is legal


def _legal_action_mask_for_single_die(board: Array, die) -> Array:
    """
    Legal action mask for a single die.
    """
    return (die == -1) * jnp.zeros(26 * 6, dtype=jnp.bool_) + (die != -1) * _legal_action_mask_for_valid_single_dice(
        board, die
    )


def _legal_action_mask_for_valid_single_dice(board: Array, die) -> Array:
    """
    Legal action mask for a single die when the die is valid.
    """
    src_indices = jnp.arange(26, dtype=jnp.int32)  # calc legal action for all src indices

    def _is_legal(idx: Array):
        action = idx * 6 + die
        legal_action_mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[action].set(_is_action_legal(board, action))
        return legal_action_mask

    legal_action_mask = jax.vmap(_is_legal)(src_indices).any(axis=0)  # (26 * 6)
    return legal_action_mask


def _get_abs_board(state: State) -> Array:
    """
    For visualization.
    """
    board: Array = state._board
    turn: Array = state._turn
    return jax.lax.cond(turn == 0, lambda: board, lambda: _flip_board(board))

# Pre-computed probability distribution for dice rolls
# First 6 indices are doubles (1,1 2,2 3,3 4,4 5,5 6,6) with probability 1/36
# Remaining indices are non-doubles with probability 2/36
_STOCHASTIC_ACTION_PROBS = jnp.array([
    # Doubles (1/36 each)
    1/36,  # 1,1
    1/36,  # 2,2
    1/36,  # 3,3
    1/36,  # 4,4
    1/36,  # 5,5
    1/36,  # 6,6
    # Non-doubles (2/36 each)
    2/36, 2/36, 2/36, 2/36, 2/36,  # 1,2 1,3 1,4 1,5 1,6
    2/36, 2/36, 2/36, 2/36,        # 2,3 2,4 2,5 2,6
    2/36, 2/36, 2/36,              # 3,4 3,5 3,6
    2/36, 2/36,                    # 4,5 4,6
    2/36,                          # 5,6
], dtype=jnp.float32)

# Pre-computed probability distribution for dice rolls
# First 6 indices are doubles (1,1 2,2 3,3 4,4 5,5 6,6) with probability 1/36
# Remaining indices are non-doubles with probability 2/36
_STOCHASTIC_SIMPLE_DOUBLES_ACTION_PROBS = jnp.array([
    # Doubles (1/36 each)
    1/6,  # 1,1
    1/6,  # 2,2
    1/6,  # 3,3
    1/6,  # 4,4
    1/6,  # 5,5
    1/6,  # 6,6
    
], dtype=jnp.float32)

# Static mapping of action indices to dice rolls
# First 6 are doubles (1,1 to 6,6)
# Remaining 15 are non-doubles in order (1,2 to 5,6)
_STOCHASTIC_DICE_MAPPING = jnp.array([
    # Doubles
    [0, 0],  # 1,1
    [1, 1],  # 2,2
    [2, 2],  # 3,3
    [3, 3],  # 4,4
    [4, 4],  # 5,5
    [5, 5],  # 6,6
    # Non-doubles
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],  # 1,2 1,3 1,4 1,5 1,6
    [1, 2], [1, 3], [1, 4], [1, 5],          # 2,3 2,4 2,5 2,6
    [2, 3], [2, 4], [2, 5],                  # 3,4 3,5 3,6
    [3, 4], [3, 5],                          # 4,5 4,6
    [4, 5],                                  # 5,6
], dtype=jnp.int32)

def action_to_str(action: Array) -> str:
    """
    Convert an action value to a human-readable string in standard backgammon notation.
    
    Args:
        action: The action value to convert (src * 6 + die)
        
    Returns:
        A string describing the action
    """
    if action < 6:  # No-op actions (src = 0)
        return f"No-op (die: {action % 6 + 1})"
    
    src_value = action // 6  # Get the raw src value before _calc_src conversion
    die_value = action % 6 + 1  # Die value (1-6)
    
    # We need to call _decompose_action to get the target
    src, die, tgt = _decompose_action(action)
    
    # In standard backgammon notation:
    # - Points are numbered 1-24
    # - Bar is referred to as "Bar"
    # - Off is referred to as "Off"
    
    # Translate internal position to standard notation
    if src_value == 0:  # No-op
        src_str = "No-op"
    elif src_value == 1:  # From bar
        src_str = "Bar"
    else:
        # Points are numbered 1-24 in standard notation
        # src_value 2-25 corresponds to points 1-24
        src_str = str(src_value - 1)
    
    if tgt == _off_idx():  # To off (bear off)
        tgt_str = "Off"
    else:
        # Adjust for 0-based indexing (points 0-23 in code → 1-24 in notation)
        tgt_str = str(tgt + 1)
    
    return f"{src_str}/{tgt_str} (die: {die_value})"

def stochastic_action_to_str(action: Array) -> str:
    """
    Convert a stochastic action (dice selection) to a human-readable string.
    
    In stochastic mode, actions 0-5 correspond to selecting a specific dice configuration
    from the probability distribution.
    
    Args:
        action: The stochastic action value (0-5)
        
    Returns:
        A string describing the dice selection
    """
    if action < 0 or action > 5:
        return f"Invalid stochastic action: {action}"
    
    dice = _STOCHASTIC_DICE_MAPPING[action]
    die1, die2 = dice[0] + 1, dice[1] + 1  # Convert from 0-based to 1-based
    
    return f"Select dice: {die1}-{die2}"

def turn_to_str(states: list[core.State], actions: list[Array]) -> str:
    """
    Convert a sequence of states and actions representing a complete turn to a human-readable string
    in standard backgammon notation.
    
    Args:
        states: List of states in the turn, from first state to last state
        actions: List of actions taken during the turn
    
    Returns:
        A string describing the complete turn in standard backgammon notation
    """
    if len(states) < 1:
        return "Empty turn"
    
    if len(actions) != len(states) - 1:
        return f"Error: {len(states)} states but {len(actions)} actions (should be {len(states)-1})"
    
    # Check if all states (except possibly the last one) have the same player
    first_player = states[0].current_player
    for i in range(1, len(states) - 1):
        if states[i].current_player != first_player:
            return f"Error: Player changed mid-turn at state {i}"
    
    # Get the dice for the turn
    dice = states[0]._dice  # type: ignore
    dice_values = [int(dice[0]) + 1, int(dice[1]) + 1]  # Convert from 0-based to 1-based
    
    # Format dice string in standard notation
    if dice_values[0] == dice_values[1]:
        dice_str = f"{dice_values[0]}-{dice_values[0]}"  # e.g., "5-5" for doubles
    else:
        dice_str = f"{dice_values[0]}-{dice_values[1]}"  # e.g., "6-4" for normal roll
    
    # Get move descriptions in standard notation
    move_strs = []
    for i, action in enumerate(actions):
        # Skip no-op actions (src = 0)
        if action // 6 != 0:
            src_value = action // 6
            die_value = action % 6 + 1
            
            # Get source and target in standard notation
            src, _, tgt = _decompose_action(action)
            
            # Format source
            if src_value == 1:  # From bar
                src_str = "Bar"
            else:
                src_str = str(src_value - 1)  # Points 1-24
            
            # Format target
            if tgt == _off_idx():  # To off (bear off)
                tgt_str = "Off"
            else:
                tgt_str = str(tgt + 1)  # Points 1-24
            
            # Standard notation doesn't include the die value in each move
            move_strs.append(f"{src_str}/{tgt_str}")
    
    # Check if the turn is complete
    is_complete = False
    if len(states) > 1:
        last_state = states[-1]
        all_dice_used = (last_state._playable_dice == -1).all()  # type: ignore
        turn_changed = (len(states) > 1 and states[-1].current_player != first_player)
        no_legal_moves = not any([(m // 6) != 0 for m in range(len(last_state.legal_action_mask)) 
                                 if last_state.legal_action_mask[m]])
        
        is_complete = all_dice_used or turn_changed or no_legal_moves
    
    # Format the final turn string in standard notation
    if not move_strs:
        return f"{dice_str}: No moves"
    
    # In standard notation, there's typically no marker for complete turns
    # but we could include one for clarity if needed
    return f"{dice_str}: {' '.join(move_strs)}"
