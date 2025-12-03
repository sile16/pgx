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

# Refactor #10: Hoist constant array allocations
# These are used repeatedly in multiple functions
_SRC_INDICES = jnp.arange(26, dtype=jnp.int32)
_NO_OP_MASK = jnp.zeros(26 * 6, dtype=jnp.bool_).at[0:6].set(True)





@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(34, dtype=jnp.int32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    _is_stochastic: Array = jnp.array(True, dtype=jnp.bool_)  # whether the state is in stochastic mode
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


def _build_lookup_tables():
    actions = jnp.arange(26 * 6, dtype=jnp.int32)
    
    # Logic derived from original _calc_src
    # src_raw: 0->NoOp, 1->Bar, 2..25->Points 0..23
    src_raw = actions // 6
    # Map raw indices to actual board positions:
    # 1 -> 24 (Bar), 0 -> -2 (No-Op), others -> src_raw - 2
    src = jnp.where(src_raw == 1, 24, jnp.where(src_raw == 0, -2, src_raw - 2))
    
    # Logic derived from original die calculation
    die = (actions % 6) + 1
    
    # Logic derived from original _calc_tgt and _from_board
    # If src is Bar (24), tgt is die - 1
    # Else, tgt is src + die (if within 0-23) or 26 (Off)
    tgt_from_bar = die - 1
    tgt_from_board = src + die
    # If move goes beyond board (>=24), it goes to Off (26)
    tgt_normal = jnp.where((tgt_from_board >= 0) & (tgt_from_board <= 23), tgt_from_board, 26)
    
    tgt = jnp.where(src >= 24, tgt_from_bar, tgt_normal)
    
    # Handle the specific No-Op case (src = -2) to keep tgt safe/consistent
    # (Though logic usually filters no-ops before checking tgt)
    tgt = jnp.where(src == -2, -2, tgt)

    return src.astype(jnp.int32), die.astype(jnp.int32), tgt.astype(jnp.int32)

# Create the constants immediately
_ACTION_SRC_LOOKUP, _ACTION_DIE_LOOKUP, _ACTION_TGT_LOOKUP = _build_lookup_tables()

class Backgammon(core.Env):
    def __init__(self, simple_doubles: bool = False, short_game: bool = False):
        super().__init__()
        self.simple_doubles = simple_doubles
        self.init_board_fn = _make_init_board_short if short_game else _make_init_board
            
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
        state = _init(key)
        state = state.replace(_board=self.init_board_fn())
        return state

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
        legal_action_mask: Array = _legal_action_mask(state._board, playable_dice, dice, played_dice_num)
        return state.replace(  # type: ignore
            current_player=state.current_player,
            _board=state._board,
            terminated=state.terminated,
            _turn=state._turn,
            _dice=dice,
            _playable_dice=playable_dice,
            _played_dice_num=played_dice_num,
            legal_action_mask=legal_action_mask,
            _is_stochastic=jnp.array(False, dtype=jnp.bool_),
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
        return "v2.3"

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
    legal_action_mask: Array = _legal_action_mask(board, playable_dice, dice, played_dice_num)
    state = State(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        _turn=turn,
        legal_action_mask=legal_action_mask,
        _is_stochastic=jnp.array(True, dtype=jnp.bool_)  #initial state is stochastic, as it requires a dice roll
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


# ==============================================================================
# Heuristic Observation Functions
# ==============================================================================
# Constants for pip count scaling
_MAX_PIP_COUNT = 375  # 15 checkers × 25 max distance (all on bar)


def _is_race(board: Array) -> Array:
    """
    Determines if the game is in a race state (no contact possible).

    A race occurs when all current player's checkers (positive) have passed
    all opponent's checkers (negative), meaning no hitting is possible.

    Returns:
        1 if race (no contact), 0 if contact possible
    """
    # Check if either player has checkers on bar - if so, not a race
    black_on_bar = board[24] > 0
    white_on_bar = board[25] < 0
    has_bar_checker = black_on_bar | white_on_bar

    # Find the minimum index where black has checkers (furthest back)
    # Use a large value (100) if no checkers exist at that position
    black_positions = jnp.where(board[:24] > 0, jnp.arange(24), 100)
    black_min_idx = jnp.min(black_positions)

    # Find the maximum index where white has checkers (furthest forward from black's view)
    # Use -1 if no checkers exist at that position
    white_positions = jnp.where(board[:24] < 0, jnp.arange(24), -1)
    white_max_idx = jnp.max(white_positions)

    # It's a race if black's furthest back checker is ahead of white's furthest forward
    # AND no one has checkers on the bar
    is_race = (black_min_idx > white_max_idx) & (~has_bar_checker)

    return is_race.astype(jnp.int32)


def _can_bear_off_current(board: Array) -> Array:
    """
    Check if current player (black/positive) can bear off.

    A player can bear off when all their checkers are on their home board
    (indices 18-23 for black) or have been borne off (index 26).

    Returns:
        1 if can bear off, 0 otherwise
    """
    # Use the existing _is_all_on_home_board function
    return _is_all_on_home_board(board).astype(jnp.int32)


def _can_bear_off_opponent(board: Array) -> Array:
    """
    Check if opponent (white/negative) can bear off.

    White can bear off when all their checkers are on their home board
    (indices 0-5 from black's perspective) or have been borne off (index 27).

    Returns:
        1 if opponent can bear off, 0 otherwise
    """
    # White's home board is indices 0-5
    # Count white checkers on home board (negative values at 0-5)
    white_home_board = board[:6]
    on_home_board = jnp.sum(jnp.minimum(jnp.maximum(-white_home_board, 0), 15))

    # Count white checkers borne off (negative at index 27, stored as negative)
    white_off = -board[27]

    # White has 15 checkers total
    # Can bear off if all remaining checkers (15 - off) are on home board
    total_remaining = 15 - white_off
    can_bear_off = (total_remaining == on_home_board)

    # Also check white has no checkers on bar
    white_on_bar = board[25] < 0

    return (can_bear_off & (~white_on_bar)).astype(jnp.int32)


def _pip_count_current(board: Array) -> Array:
    """
    Calculate pip count for current player (black/positive).

    Pip count = sum of (distance to bear off) for each checker.
    - For checkers at index i (0-23): distance = 24 - i
    - For checkers on bar (index 24): distance = 25

    Returns:
        Total pip count for current player
    """
    # Pip distances for board positions 0-23
    distances = 24 - jnp.arange(24, dtype=jnp.int32)

    # Count only positive values (black checkers)
    black_counts = jnp.maximum(board[:24], 0)
    board_pips = jnp.sum(black_counts * distances)

    # Add bar checkers (distance = 25)
    bar_pips = jnp.maximum(board[24], 0) * 25

    return board_pips + bar_pips


def _pip_count_opponent(board: Array) -> Array:
    """
    Calculate pip count for opponent (white/negative).

    Pip count = sum of (distance to bear off) for each checker.
    White moves from high indices to low, bearing off at index -1 (conceptually).
    - For checkers at index i (0-23): distance = i + 1
    - For checkers on bar (index 25): distance = 25

    Returns:
        Total pip count for opponent
    """
    # Pip distances for white: index + 1 (white moves toward index 0)
    distances = jnp.arange(24, dtype=jnp.int32) + 1

    # Count only negative values (white checkers), take absolute value
    white_counts = jnp.maximum(-board[:24], 0)
    board_pips = jnp.sum(white_counts * distances)

    # Add bar checkers (distance = 25)
    bar_pips = jnp.maximum(-board[25], 0) * 25

    return board_pips + bar_pips


def _pip_count_differential_scaled(board: Array) -> Array:
    """
    Calculate scaled pip count differential.

    Positive value = current player (black) is ahead (lower pip count).
    Negative value = opponent (white) is ahead.

    Scaled to [-1, 1] range by dividing by max possible pip count (375).

    Returns:
        Scaled pip differential in range [-1, 1]
    """
    current_pips = _pip_count_current(board)
    opponent_pips = _pip_count_opponent(board)

    # Positive differential = current player is ahead (fewer pips)
    differential = opponent_pips - current_pips

    # Scale to [-1, 1]
    return differential.astype(jnp.float32) / _MAX_PIP_COUNT


def _observe_with_heuristics(state: State) -> Array:
    """
    Return observation for current player with additional heuristic features.

    Observation structure (38 elements):
    - [0:28]  Board state (28 elements)
    - [28:34] Playable dice count (6 elements)
    - [34]    Race flag (1 = race, 0 = contact possible)
    - [35]    Current player can bear off (1 = yes, 0 = no)
    - [36]    Opponent can bear off (1 = yes, 0 = no)
    - [37]    Scaled pip count differential (positive = current ahead)

    Returns:
        38-element observation array
    """
    board: Array = state._board

    # Basic observation components
    playable_dice_count_vec: Array = _to_playable_dice_count(state._playable_dice)

    # Heuristic features
    race_flag = _is_race(board)
    bear_off_current = _can_bear_off_current(board)
    bear_off_opponent = _can_bear_off_opponent(board)
    pip_differential = _pip_count_differential_scaled(board)

    # Concatenate all features
    return jnp.concatenate([
        board,                                    # 28 elements
        playable_dice_count_vec,                  # 6 elements
        jnp.array([race_flag], dtype=jnp.float32),          # 1 element
        jnp.array([bear_off_current], dtype=jnp.float32),   # 1 element
        jnp.array([bear_off_opponent], dtype=jnp.float32),  # 1 element
        jnp.array([pip_differential], dtype=jnp.float32),   # 1 element
    ])


def _blot_board(board: Array) -> Array:
    """
    Extract blot positions for points 0-23.

    A blot is a single checker that can be hit by the opponent.

    Returns:
        24-element array: +1 for current player blot, -1 for opponent blot, 0 otherwise
    """
    points = board[:24].astype(jnp.float32)
    abs_points = jnp.abs(points)
    sign = jnp.sign(points)
    # Blot when exactly 1 checker, sign gives direction
    return (abs_points == 1).astype(jnp.float32) * sign


def _blocker_board(board: Array) -> Array:
    """
    Extract blocker positions for points 0-23.

    A blocker is 2 or more checkers that form a "made point" blocking opponent movement.

    Returns:
        24-element array:
        - +0.5 for 2 checkers, +1 for 3+ checkers (current player)
        - -0.5 for 2 checkers, -1 for 3+ checkers (opponent)
        - 0 otherwise
    """
    points = board[:24].astype(jnp.float32)
    abs_points = jnp.abs(points)
    sign = jnp.sign(points)
    # 2 comparisons instead of 4: use abs and sign
    is_2 = (abs_points == 2).astype(jnp.float32)
    is_3plus = (abs_points >= 3).astype(jnp.float32)
    return sign * (is_2 * 0.5 + is_3plus)


def _observe_full(state: State) -> Array:
    """
    Return scaled observation with all features (86 elements).

    All values normalized to [-1, 1] range for ML.

    Observation structure:
    - [0:28]   Scaled board (raw / 15)
    - [28:34]  Scaled dice counts (raw / 4)
    - [34]     Race flag (1 = race, 0 = contact possible)
    - [35]     Current player can bear off (1 = yes, 0 = no)
    - [36]     Opponent can bear off (1 = yes, 0 = no)
    - [37]     Scaled pip count differential (positive = current ahead)
    - [38:62]  Blot board (+1 current blot, -1 opponent blot)
    - [62:86]  Blocker board (±0.5 for 2, ±1 for 3+)

    Returns:
        86-element observation array with all values in [-1, 1]
    """
    board = state._board
    board_f = board.astype(jnp.float32)
    points = board_f[:24]

    # Pre-compute shared values for blot/blocker (optimization)
    abs_points = jnp.abs(points)
    sign = jnp.sign(points)

    # Scaled board (28 elements) - divide by 15
    scaled_board = board_f / 15.0

    # Scaled dice (6 elements) - divide by 4
    dice_counts = _to_playable_dice_count(state._playable_dice)
    scaled_dice = dice_counts.astype(jnp.float32) / 4.0

    # Heuristics as single array (4 elements) - reduces concatenations
    heuristics = jnp.array([
        _is_race(board).astype(jnp.float32),
        _can_bear_off_current(board).astype(jnp.float32),
        _can_bear_off_opponent(board).astype(jnp.float32),
        _pip_count_differential_scaled(board),
    ])

    # Inline blot/blocker using pre-computed abs/sign (48 elements)
    blots = (abs_points == 1).astype(jnp.float32) * sign
    blockers = sign * ((abs_points == 2).astype(jnp.float32) * 0.5 + (abs_points >= 3).astype(jnp.float32))

    # 5 concatenations instead of 8
    return jnp.concatenate([scaled_board, scaled_dice, heuristics, blots, blockers])


def _to_playable_dice_count(playable_dice: Array) -> Array:
    """
    Return 6 dim vec which represents the number of playable die
    Examples
    Playable dice: 2, 3
    Return: [0, 1, 1, 0, 0, 0]

    Playable dice: 4, 4, 4, 4
    Return: [0, 0, 0, 0, 4, 0]

    Refactor #6: Use histogram-style counting instead of vmap + tile.
    """
    # Create mask for valid dice (not -1)
    valid_mask = playable_dice != -1

    # Count occurrences of each die value (0-5)
    # For each die value i, count how many times it appears in playable_dice
    die_values = jnp.arange(6, dtype=jnp.int32)

    def count_die(die_val):
        return ((playable_dice == die_val) & valid_mask).sum()

    return jax.vmap(count_die)(die_values).astype(jnp.int32)


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
    legal_action_mask: Array = _legal_action_mask(board, playable_dice, state._dice, played_dice_num)
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

def _make_init_board_short() -> Array:
    """
    Initialize the board based on black's perspective.
    This board is further along in the game, so game durations are shorter.
    But it still has all the components of a normal backgammon board.
    """
    board: Array = jnp.array([0, -1, -3, 0, 2, -3, 0, -3, -2, 0, 0, -1, 1, 0, 0, 2, 3, 0, 3, -2, 0, 3, 1, 0, 0, 0, 0, 0], dtype=jnp.int32)  # type: ignore
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
    legal_action_mask: Array = _legal_action_mask(board, playable_dice, dice, played_dice_num)
    return state.replace(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
        _is_stochastic=jnp.array(True, dtype=jnp.bool_) , #after a player change it's a stochastic state
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
    Set playable dice array. -1 for empty slots.
    - Doubles: [d, d, d, d]
    - Non-doubles: [d1, d2, -1, -1]

    Refactor #8: Use jnp.where instead of arithmetic with array literals.
    """
    is_doubles = dice[0] == dice[1]
    doubles_result = jnp.full(4, dice[0], dtype=jnp.int32)
    non_doubles_result = jnp.array([dice[0], dice[1], -1, -1], dtype=jnp.int32)
    return jnp.where(is_doubles, doubles_result, non_doubles_result)


def _update_playable_dice(
    playable_dice: Array,
    played_dice_num: Array,
    dice: Array,
    action: Array,
) -> Array:
    """
    Update playable_dice after a move.
    - For doubles: mark slot (3 - played_dice_num) as used (-1)
    - For non-doubles: find and mark the first slot matching the played die

    Refactor #7: Simplified non-doubles case using jnp.where instead of vmap+tile.
    """
    played_die = action % 6
    is_doubles = dice[0] == dice[1]

    # For doubles: mark the next slot as used (slots filled from index 3 down to 0)
    doubles_result = playable_dice.at[3 - played_dice_num].set(-1)

    # For non-doubles: mark the first matching die as used
    # playable_dice is [die1, die2, -1, -1] for non-doubles
    # We need to set the first slot matching played_die to -1
    matches = playable_dice == played_die
    # Find first match index (0 or 1 for non-doubles)
    first_match_idx = jnp.argmax(matches)
    non_doubles_result = playable_dice.at[first_match_idx].set(-1)

    return jnp.where(is_doubles, doubles_result, non_doubles_result)


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


def _decompose_action(action: Array):
    """
    Decompose action to src, die, tgt using pre-computed lookup tables.
    Refactor #3: Replaced math/branching with O(1) array access.
    """
    return (
        _ACTION_SRC_LOOKUP[action],
        _ACTION_DIE_LOOKUP[action],
        _ACTION_TGT_LOOKUP[action]
    )


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

def _get_valid_sequence_mask(board: Array, die_first: int, die_second: int) -> Array:
    """
    Calculates a mask of valid moves for 'die_first' that ALSO allow 'die_second'
    to be played afterwards.

    Refactor #4: Focused Candidate Search
    Optimization: Only process the 26 actions that correspond to 'die_first'.
    Previously, we processed all 156 actions, wasting 83% of compute on
    actions that didn't match the die.
    """
    # 1. Identify candidates: Only check sources 0..25 combined with die_first
    # Shape: (26,) - specific action indices
    candidate_actions = _SRC_INDICES * 6 + (die_first - 1)

    # 2. Check legality of these specific actions on current board
    # Shape: (26,)
    is_legal_first_move = jax.vmap(_is_action_legal, in_axes=(None, 0))(board, candidate_actions)

    # 3. Speculate: Apply ONLY the 26 candidate actions
    # Shape: (26, 28)
    # This reduces the memory/compute footprint of the next step by 6x
    next_boards = jax.vmap(_move, in_axes=(None, 0))(board, candidate_actions)

    # 4. Check future legality: Is die_second playable on these new boards?
    # We use a localized helper to avoid generating the full 156-bool mask
    def _check_any_move(b, d):
        # Check if *any* source allows moving 'd'
        # We construct the 26 possible actions for die 'd'
        acts = _SRC_INDICES * 6 + (d - 1)
        return jax.vmap(_is_action_legal, in_axes=(None, 0))(b, acts).any()

    # Shape: (26,)
    can_play_second_die = jax.vmap(_check_any_move, in_axes=(0, None))(
        next_boards, die_second
    )

    # 5. Combine: valid if first move is legal AND second move is possible
    # Shape: (26,)
    valid_candidates = is_legal_first_move & can_play_second_die

    # 6. Scatter: Map the 26 results back to the full 156-action space
    # Initialize full mask as False, then update the specific indices
    full_mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
    return full_mask.at[candidate_actions].set(valid_candidates)

def _compute_two_dice_masks(board: Array, die1: int, die2: int):
    """
    Computes valid sequence masks for both dice orderings.

    Refactor #9: Fused computation - compute both masks once and return them.
    Previously, _can_play_two_dice and _get_forced_full_move_mask would
    each compute these same masks independently, resulting in 4 calls to
    _get_valid_sequence_mask instead of 2.

    Returns:
        Tuple of (mask_d1_then_d2, mask_d2_then_d1, can_play_both)
    """
    # Check sequence die1 -> die2
    mask_d1_then_d2 = _get_valid_sequence_mask(board, die1, die2)

    # Check sequence die2 -> die1
    mask_d2_then_d1 = _get_valid_sequence_mask(board, die2, die1)

    # Check if any valid sequence exists in either direction
    can_play_both = mask_d1_then_d2.any() | mask_d2_then_d1.any()

    return mask_d1_then_d2, mask_d2_then_d1, can_play_both


def _get_forced_single_move_mask(board: Array, die1: int, die2: int) -> Array:
    """Returns a mask of legal moves when only one die can be played.
    The higher die must be chosen if it's playable.

    Args:
        board: The current board state.
        die1: The first die roll (1-6).
        die2: The second die roll (1-6).

    Returns:
        A boolean mask of legal actions.
    """
    d_h = jnp.maximum(die1, die2)
    d_l = jnp.minimum(die1, die2)
    
    mask_h = _legal_action_mask_for_valid_single_dice(board, d_h - 1)
    can_play_h = mask_h.any()
    
    mask_l = _legal_action_mask_for_valid_single_dice(board, d_l - 1)
    
    return jax.lax.cond(
        can_play_h,
        lambda: mask_h,
        lambda: mask_l,
    )


def _apply_special_backgammon_rules(board: Array, turn_dice: Array) -> Array:
    """Computes the legal action mask considering two special backgammon rules for non-doubles:
    1. If you can play both dice, you must. A move that makes the second die unplayable is illegal.
    2. If you can only play one die, you must play the higher one if possible.

    This function should only be called at the start of a non-double turn.

    Refactor #9: Fused mask computation - compute masks once and reuse.
    Previously called _can_play_two_dice then _get_forced_full_move_mask,
    each of which computed the same two masks independently.
    """
    d1_0idx, d2_0idx = turn_dice[0], turn_dice[1]
    d1 = d1_0idx + 1
    d2 = d2_0idx + 1

    # Compute both sequence masks once (replaces 4 calls with 2)
    mask_d1_then_d2, mask_d2_then_d1, can_play_both = _compute_two_dice_masks(board, d1, d2)

    # Combine the masks for the forced full move case
    forced_full_move_mask = mask_d1_then_d2 | mask_d2_then_d1

    return jax.lax.cond(
        can_play_both,
        # Rule 1: Forced full move - use pre-computed combined mask
        lambda: forced_full_move_mask,
        # Rule 2: Forced single (higher) move
        lambda: _get_forced_single_move_mask(board, d1, d2),
    )


def _legal_action_mask(board: Array, playable_dice: Array, turn_dice: Array, played_dice_num: Array) -> Array:
    # 1. OPTIMIZATION: Only look at the first 2 dice slots.
    #    - Non-doubles: slots 2 and 3 are always -1.
    #    - Doubles: slots 2 and 3 are redundant duplicates of 0 and 1.
    #    This minimizes the input size for unique/vmap.
    dice_to_check = playable_dice[:2]
    
    # 2. Deduplication: Identify unique dice values in those first 2 slots.
    #    Returns at most 2 values (e.g., [3, 5] or [4, -1] or [-1, -1]).
    unique_dice = jnp.unique(dice_to_check, size=2, fill_value=-1)

    # 3. Compute masks only for these unique dice.
    #    Shape: (2, 156)
    unique_masks = jax.vmap(partial(_legal_action_mask_for_single_die, board=board))(die=unique_dice)

    # 4. Combine: Logical OR to see if any available die allows a move.
    simple_legal_mask = unique_masks.any(axis=0)

    # Apply special rules only at the start of a turn (played_dice_num == 0) for non-doubles.
    is_start_of_turn = (played_dice_num == 0)
    is_nondoubles = (turn_dice[0] != turn_dice[1])

    def apply_rules_fn():
        return _apply_special_backgammon_rules(board, turn_dice)

    def keep_simple_mask_fn():
        return simple_legal_mask

    final_mask = jax.lax.cond(
        is_start_of_turn & is_nondoubles,
        apply_rules_fn,
        keep_simple_mask_fn
    )

    # If no moves are possible after filtering, allow no-op.
    legal_action_exists = final_mask.any()
    return jax.lax.cond(
        legal_action_exists,
        lambda: final_mask,
        lambda: _NO_OP_MASK
    )


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

    Refactor #5: Direct scatter instead of creating 26 full masks.
    Previously created (26, 156) intermediate array, now creates (26,) + scatter.
    """
    actions = _SRC_INDICES * 6 + die

    # Check legality for all 26 actions at once - Shape: (26,)
    is_legal = jax.vmap(_is_action_legal, in_axes=(None, 0))(board, actions)

    # Scatter results directly into 156-element mask
    mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
    return mask.at[actions].set(is_legal)


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
# Simple double only mode
_STOCHASTIC_SIMPLE_DOUBLES_ACTION_PROBS = jnp.array([
    # Doubles (1/36 each)
    1/6,  # 1,1
    1/6,  # 2,2
    1/6,  # 3,3
    1/6,  # 4,4
    1/6,  # 5,5
    1/6,  # 6,6
    # make all other dice rolls 0 probability
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
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
    if action < 0 or action >= _STOCHASTIC_DICE_MAPPING.shape[0]:
        return f"Invalid stochastic action: {action}"
    
    dice = _STOCHASTIC_DICE_MAPPING[action]
    die1, die2 = dice[0] + 1, dice[1] + 1  # Convert from 0-based to 1-based
    
    return f"Rolled: {die1}-{die2}"

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