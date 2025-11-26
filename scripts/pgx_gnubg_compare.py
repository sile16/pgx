import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import gnubg
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgx import backgammon
from tools.gnubg_bridge import (
    apply_gnubg_steps,
    apply_pgx_actions,
    enumerate_pgx_moves,
    flip_pgx_board,
    gnubg_first_steps,
    gnubg_legal_moves,
    gnubg_to_pgx_board,
    pgx_single_steps,
    pgx_to_gnubg_board,
    swap_gnubg_perspective,
)


Matchup = Tuple[str, str]  # policy for player 0, policy for player 1


def roll_non_double(rng: np.random.Generator) -> Tuple[int, int]:
    while True:
        dice = rng.integers(1, 7, size=2)
        if dice[0] != dice[1]:
            return int(dice[0]), int(dice[1])


def select_policy_move(
    policy: str,
    gnubg_board: Sequence[Sequence[int]],
    dice: Tuple[int, int],
    legal_moves: List[Tuple[Tuple[int, int], ...]],
    rng: np.random.Generator,
) -> Tuple[Tuple[int, int], ...]:
    if not legal_moves:
        return tuple()

    if policy == "random":
        return tuple(legal_moves[rng.integers(0, len(legal_moves))])
    if policy == "gnubg":
        from tools.gnubg_bridge import full_move_to_steps

        chosen = gnubg.best_move(gnubg_board, dice[0], dice[1])
        return full_move_to_steps(chosen)
    raise ValueError(f"Unknown policy: {policy}")


def compare_move_sets(pgx_moves, gnubg_moves) -> Tuple[bool, List[Tuple]]:
    pgx_set = set(pgx_moves)
    gnubg_set = set(gnubg_moves)
    if pgx_set == gnubg_set:
        return True, []
    missing = sorted(pgx_set.symmetric_difference(gnubg_set))
    return False, missing


def _format_moves(moves: Sequence[Tuple[Tuple[int, int], ...]]) -> str:
    """Human-readable move list."""
    def fmt_step(step: Tuple[int, int]) -> str:
        return f"{step[0]}->{step[1]}"

    def fmt_move(move: Tuple[Tuple[int, int], ...]) -> str:
        return "[" + ", ".join(fmt_step(s) for s in move) + "]"

    sorted_moves = sorted(moves)
    return ", ".join(fmt_move(m) for m in sorted_moves)


def _format_pgx_board(board: Sequence[int]) -> str:
    """Simple human-readable board for the current player perspective."""
    top = []
    bottom = []
    for i in range(12):
        pt = 24 - i
        val = board[i]
        top.append(f"{pt}:{val:+d}")
    for i in range(12, 24):
        pt = 24 - i
        val = board[i]
        bottom.append(f"{pt}:{val:+d}")
    bar_cur = int(max(board[24], 0))
    bar_opp = int(max(-board[25], 0))
    off_cur = int(board[26])
    off_opp = int(-board[27])
    parts = [
        "top 24->13: " + " ".join(top),
        "bot 12->1 : " + " ".join(bottom),
        f"bar cur/opp: {bar_cur}/{bar_opp}  off cur/opp: {off_cur}/{off_opp}",
    ]
    return "\n".join(parts)


def _format_steps(steps: Sequence[Tuple[int, int]]) -> str:
    return ", ".join(f"{a}->{b}" for a, b in sorted(steps))


def _format_board_key(key: Tuple[int, ...]) -> str:
    return "[" + ",".join(str(x) for x in key) + "]"


def _format_missing_states(
    keys: Sequence[Tuple[int, ...]],
    state_to_moves: dict,
) -> str:
    """Render missing end states with the move sequences that reach them."""
    lines = []
    for key in keys:
        moves = state_to_moves.get(key, [])
        lines.append(f"{_format_board_key(key)} -> {_format_moves(moves)}")
    return "\n".join(lines)


def play_game(
    matchup: Matchup,
    rng: np.random.Generator,
    log_lines: List[str],
    game_idx: int,
    max_turns: int,
) -> None:
    pgx_board = np.array(backgammon._make_init_board())
    dice = roll_non_double(rng)
    current_player = 0 if dice[0] > dice[1] else 1
    if current_player == 1:
        pgx_board = flip_pgx_board(pgx_board)

    gnubg_board = pgx_to_gnubg_board(pgx_board)

    for turn in range(max_turns):
        policies = matchup
        policy = policies[current_player]

        # Full sequences for driving the game
        gnubg_moves = gnubg_legal_moves(gnubg_board, dice)
        pgx_moves_map = enumerate_pgx_moves(pgx_board, dice)
        pgx_moves = list(pgx_moves_map.keys())

        # End-state comparison (sets of resulting boards)
        pgx_end_states = {}
        for seq, acts in pgx_moves_map.items():
            end_board = apply_pgx_actions(pgx_board, acts)
            key = tuple(int(x) for x in end_board.tolist())
            pgx_end_states.setdefault(key, []).append(seq)

        gnubg_end_states = {}
        for seq in gnubg_moves:
            end_board = gnubg_to_pgx_board(apply_gnubg_steps(gnubg_board, seq))
            key = tuple(int(x) for x in end_board.tolist())
            gnubg_end_states.setdefault(key, []).append(seq)

        pgx_keys = set(pgx_end_states.keys())
        gnubg_keys = set(gnubg_end_states.keys())
        missing_in_gnubg = sorted(pgx_keys - gnubg_keys)
        missing_in_pgx = sorted(gnubg_keys - pgx_keys)

        no_moves_gnubg = len(gnubg_moves) == 0
        no_moves_pgx = (not pgx_moves) or (set(pgx_moves) == {tuple()})

        if (no_moves_gnubg or no_moves_pgx) and (
            pgx_keys != gnubg_keys or set(pgx_moves) != set(gnubg_moves)
        ):
            pgx_vis = _format_pgx_board(pgx_board)
            gnubg_vis = _format_pgx_board(gnubg_to_pgx_board(gnubg_board))
            log_lines.append(
                f"[game {game_idx} turn {turn}] ZERO-MOVE DEBUG player {current_player} dice {dice}\n"
                f"pgx board:\n{pgx_vis}\n"
                f"gnubg board:\n{gnubg_vis}\n"
                f"pgx moves ({len(pgx_moves)}): {_format_moves(pgx_moves)}\n"
                f"gnubg moves ({len(gnubg_moves)}): {_format_moves(gnubg_moves)}\n"
                f"pgx end states ({len(pgx_keys)}): {[ _format_board_key(k) for k in pgx_keys ]}\n"
                f"gnubg end states ({len(gnubg_keys)}): {[ _format_board_key(k) for k in gnubg_keys ]}"
            )

        # Single-step comparison (still useful as supporting detail)
        gnubg_steps = gnubg_first_steps(gnubg_board, dice)
        pgx_steps = pgx_single_steps(pgx_board, dice)
        step_same, step_delta = compare_move_sets(pgx_steps, gnubg_steps)

        if missing_in_gnubg or missing_in_pgx:
            missing_in_gnubg_moves = _format_missing_states(missing_in_gnubg, pgx_end_states)
            missing_in_pgx_moves = _format_missing_states(missing_in_pgx, gnubg_end_states)
            pgx_vis = _format_pgx_board(pgx_board)
            gnubg_vis = _format_pgx_board(gnubg_to_pgx_board(gnubg_board))
            log_lines.append(
                f"[game {game_idx} turn {turn}] player {current_player} dice {dice}\n"
                f"pgx board:\n{pgx_vis}\n"
                f"gnubg board:\n{gnubg_vis}\n"
                f"pgx end states ({len(pgx_keys)}): {[ _format_board_key(k) for k in pgx_keys ]}\n"
                f"gnubg end states ({len(gnubg_keys)}): {[ _format_board_key(k) for k in gnubg_keys ]}\n"
                f"missing in gnubg ({len(missing_in_gnubg)}): {[ _format_board_key(k) for k in missing_in_gnubg ]}\n"
                f"moves to missing gnubg states:\n{missing_in_gnubg_moves}\n"
                f"missing in pgx ({len(missing_in_pgx)}): {[ _format_board_key(k) for k in missing_in_pgx ]}\n"
                f"moves to missing pgx states:\n{missing_in_pgx_moves}\n"
                f"pgx single ({len(pgx_steps)}): {_format_steps(pgx_steps)}\n"
                f"gnubg single ({len(gnubg_steps)}): {_format_steps(gnubg_steps)}\n"
                f"delta singles ({len(step_delta)}): {_format_steps(step_delta)}"
            )

        chosen_steps = select_policy_move(policy, gnubg_board, dice, gnubg_moves, rng)
        if chosen_steps and chosen_steps not in pgx_moves_map:
            log_lines.append(
                f"[game {game_idx} turn {turn}] player {current_player} chose move not in pgx set: {chosen_steps}"
            )
        actions = pgx_moves_map.get(chosen_steps, [])

        pgx_board_after = apply_pgx_actions(pgx_board, actions)
        gnubg_board_after = apply_gnubg_steps(gnubg_board, chosen_steps)
        translated = gnubg_to_pgx_board(gnubg_board_after)

        if not np.array_equal(pgx_board_after, translated):
            log_lines.append(
                f"[game {game_idx} turn {turn}] board diverged after move; pgx {pgx_board_after.tolist()} gnubg {translated.tolist()}"
            )

        if int(pgx_board_after[26]) == 15:
            return  # game over, current player won

        # Prepare for next turn
        pgx_board = flip_pgx_board(pgx_board_after)
        gnubg_board = swap_gnubg_perspective(gnubg_board_after)
        current_player = 1 - current_player
        dice = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))


def main():
    parser = argparse.ArgumentParser(description="Compare pgx backgammon with GNU Backgammon.")
    parser.add_argument("--games", type=int, default=1000, help="Games per matchup")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--log", type=Path, default=Path("logs/pgx_gnubg_diff.log"))
    parser.add_argument("--max-turns", type=int, default=512, help="Safety cap per game")
    parser.add_argument(
        "--matchups",
        type=str,
        nargs="*",
        default=["rand_rand", "rand_gnubg", "gnubg_gnubg"],
        help="Subset of matchups to run",
    )
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)

    matchups = {
        "rand_rand": ("random", "random"),
        "rand_gnubg": ("random", "gnubg"),
        "gnubg_gnubg": ("gnubg", "gnubg"),
    }

    log_lines: List[str] = []
    for name, policies in matchups.items():
        if name not in args.matchups:
            continue
        for g in range(args.games):
            rng = np.random.default_rng(args.seed + g)
            play_game(policies, rng, log_lines, g, args.max_turns)
        print(f"Finished {args.games} games for {name}")

    args.log.write_text("\n".join(log_lines))
    print(f"Wrote diff log to {args.log} ({len(log_lines)} entries)")


if __name__ == "__main__":
    main()
