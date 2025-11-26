from typing import Dict, Iterable, List, Sequence, Tuple, Union

import jax
import gnubg
import jax.numpy as jnp
import numpy as np

from pgx import backgammon

# Step is expressed in GNU Backgammon point numbers:
# 25 = bar, 24..1 = points from farthest to home, 0 = borne off.
Step = Tuple[int, int]

_jit_legal_mask = jax.jit(backgammon._legal_action_mask)
_jit_move = jax.jit(backgammon._move)
_jit_update_dice = jax.jit(backgammon._update_playable_dice)


def pgx_to_gnubg_board(pgx_board: np.ndarray) -> List[List[int]]:
    """Convert a pgx board (current-player perspective, length 28) to
    a GNU Backgammon board [opponent, current].
    """
    current = [0] * 25
    opponent = [0] * 25

    for idx in range(24):
        val = int(pgx_board[idx])
        if val > 0:
            current[23 - idx] = val
        elif val < 0:
            opponent[idx] = -val

    current[24] = int(max(pgx_board[24], 0))  # current bar
    opponent[24] = int(max(-pgx_board[25], 0))  # opponent bar
    return [opponent, current]


def gnubg_to_pgx_board(board: Sequence[Sequence[int]]) -> np.ndarray:
    """Convert a GNU Backgammon board back into pgx's 28-slot array."""
    opponent, current = board
    pgx_board = np.zeros(28, dtype=np.int32)

    # Current player pieces (positive)
    for idx, count in enumerate(current[:24]):
        if count:
            pgx_board[23 - idx] = count
    pgx_board[24] = current[24]  # bar
    pgx_board[26] = 15 - sum(current)  # off

    # Opponent pieces (negative)
    for idx, count in enumerate(opponent[:24]):
        if count:
            pgx_board[idx] = -count
    pgx_board[25] = -opponent[24]  # opponent bar (negative)
    pgx_board[27] = -(15 - sum(opponent))  # opponent off (negative)
    return pgx_board


def gnubg_legal_moves(board: Sequence[Sequence[int]], dice: Tuple[int, int]) -> List[Tuple[Step, ...]]:
    """List legal move sequences from GNU Backgammon for the given dice."""
    moves = gnubg.moves(board, int(dice[0]), int(dice[1]), 1)
    if not moves:
        return [tuple()]  # explicit no-op when gnubg reports no legal moves
    return [full_move_to_steps(steps) for _, steps in moves]


def gnubg_first_steps(board: Sequence[Sequence[int]], dice: Tuple[int, int]) -> List[Step]:
    """Return unique first steps from GNU Backgammon move list."""
    seqs = gnubg_legal_moves(board, dice)
    seen = []
    for mv in seqs:
        if not mv:
            continue
        step = mv[0]
        if step not in seen:
            seen.append(step)
    return seen


def full_move_to_steps(move: Union[Tuple[int, ...], Tuple[Step, ...]]) -> Tuple[Step, ...]:
    """Normalize a GNU Backgammon move into a tuple of (from, to) pip steps.

    The gnubg API sometimes yields ((24, 23), (13, 11)) or a flat tuple like
    (24, 23, 13, 11). This helper makes sure we always return a tuple of
    2-tuples.
    """
    if not move:
        return tuple()
    if isinstance(move[0], tuple):  # already steps
        return tuple(move)  # type: ignore[arg-type]
    # Flattened form
    flat = list(move)  # type: ignore[arg-type]
    if len(flat) % 2 != 0:
        raise ValueError(f"Unexpected flat move length: {move}")
    steps: List[Step] = []
    for i in range(0, len(flat), 2):
        steps.append((int(flat[i]), int(flat[i + 1])))
    return tuple(steps)


def _action_to_step(action: int) -> Step:
    src, _, tgt = backgammon._decompose_action(jnp.int32(action))
    src = int(src)
    tgt = int(tgt)
    if src < 0:  # no-op
        return (-1, -1)
    from_point = 25 if src == 24 else 24 - src
    to_point = 0 if tgt == 26 else 24 - tgt
    return from_point, to_point


def enumerate_pgx_moves(board: np.ndarray, dice: Tuple[int, int]) -> Dict[Tuple[Step, ...], List[int]]:
    """Enumerate all full-move sequences pgx considers legal for the dice."""
    dice_arr = jnp.array([dice[0] - 1, dice[1] - 1], dtype=jnp.int32)
    initial_playable = backgammon._set_playable_dice(dice_arr)
    sequences: Dict[Tuple[Step, ...], List[int]] = {}

    def dfs(cur_board, playable, played_count: int, actions: List[int]):
        mask = _jit_legal_mask(cur_board, playable, dice_arr, jnp.int32(played_count))
        mask_np = np.array(mask)

        no_op_only = mask_np.any() and mask_np[:6].all() and mask_np.sum() == 6
        if (not mask_np.any()) or no_op_only:
            steps = []
            for a in actions:
                step = _action_to_step(a)
                if step != (-1, -1):
                    steps.append(step)
            step_key = tuple(steps)
            sequences.setdefault(step_key, actions.copy())
            return

        legal_actions = [int(a) for a in np.where(mask_np)[0] if a // 6 != 0]
        if not legal_actions:
            steps = []
            for a in actions:
                step = _action_to_step(a)
                if step != (-1, -1):
                    steps.append(step)
            sequences.setdefault(tuple(steps), actions.copy())
            return

        for act in legal_actions:
            next_board = _jit_move(cur_board, jnp.int32(act))
            next_playable = _jit_update_dice(
                playable, jnp.int32(played_count), dice_arr, jnp.int32(act)
            )
            dfs(next_board, next_playable, played_count + 1, actions + [act])

    dfs(jnp.array(board, dtype=jnp.int32), initial_playable, 0, [])
    return sequences


def pgx_single_steps(board: np.ndarray, dice: Tuple[int, int]) -> List[Step]:
    """Return unique single steps pgx allows at the start of the turn."""
    dice_arr = jnp.array([dice[0] - 1, dice[1] - 1], dtype=jnp.int32)
    playable = backgammon._set_playable_dice(dice_arr)
    mask = _jit_legal_mask(jnp.array(board, dtype=jnp.int32), playable, dice_arr, jnp.int32(0))
    steps = []
    for idx, allowed in enumerate(np.array(mask)):
        if not allowed:
            continue
        if idx // 6 == 0:  # no-op group
            continue
        step = _action_to_step(idx)
        if step not in steps and step != (-1, -1):
            steps.append(step)
    return steps


def apply_gnubg_steps(board: Sequence[Sequence[int]], steps: Iterable[Step]) -> List[List[int]]:
    """Apply a GNU Backgammon step sequence, returning a new board."""
    opponent = list(board[0])
    current = list(board[1])

    for from_pt, to_pt in steps:
        src_idx = 24 if from_pt == 25 else from_pt - 1
        if current[src_idx] <= 0:
            raise ValueError(f"No checker to move from {from_pt} on current board {current}")
        current[src_idx] -= 1

        if to_pt == 0:
            continue  # borne off

        dest_idx = to_pt - 1
        opp_idx = 24 - to_pt
        if opponent[opp_idx] == 1:
            opponent[opp_idx] = 0
            opponent[24] += 1
        elif opponent[opp_idx] > 1:
            raise ValueError(f"Illegal hit attempt onto blocked point {to_pt}")

        current[dest_idx] += 1

    return [opponent, current]


def apply_pgx_actions(board: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    """Apply a list of pgx action indices to a board."""
    cur = jnp.array(board, dtype=jnp.int32)
    for act in actions:
        cur = backgammon._move(cur, jnp.int32(act))
    return np.array(cur)


def flip_pgx_board(board: np.ndarray) -> np.ndarray:
    """Flip perspective for the next player."""
    return np.array(backgammon._flip_board(jnp.array(board, dtype=jnp.int32)))


def swap_gnubg_perspective(board: Sequence[Sequence[int]]) -> List[List[int]]:
    """Swap GNU Backgammon board so side-to-move is in slot 1."""
    return [list(board[1]), list(board[0])]
