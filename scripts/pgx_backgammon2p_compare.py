import argparse
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from pgx.backgammon import Backgammon, _STOCHASTIC_ACTION_PROBS, _STOCHASTIC_DICE_MAPPING, _decision_step as bg_step
from pgx.backgammon2p import Backgammon2P, _decision_step as bg2_step


BoardKey = Tuple[int, ...]
EndKey = Tuple[BoardKey, int, int]


def _board_key(board: jnp.ndarray) -> BoardKey:
    return tuple(np.asarray(board, dtype=np.int32).tolist())


def _state_key_bg(state) -> Tuple[BoardKey, Tuple[int, ...], int, int, int]:
    return (
        _board_key(state._board),
        tuple(np.asarray(state._playable_dice, dtype=np.int32).tolist()),
        int(state._played_dice_num),
        int(state.current_player),
        int(state._turn),
    )


def _state_key_bg2(state) -> Tuple[BoardKey, int, int, int]:
    return (
        _board_key(state._board),
        int(state._remaining_actions),
        int(state.current_player),
        int(state._turn),
    )


def _end_key(state) -> EndKey:
    return (_board_key(state._board), int(state.current_player), int(state._turn))


def _enumerate_end_states(state, step_fn, state_key_fn) -> Dict[EndKey, object]:
    cache: Dict[Tuple, Dict[EndKey, object]] = {}

    def collect(s) -> Dict[EndKey, object]:
        if bool(s.terminated) or bool(s._is_stochastic):
            return {_end_key(s): s}
        key = state_key_fn(s)
        if key in cache:
            return cache[key]
        mask = np.asarray(s.legal_action_mask, dtype=bool)
        actions = np.flatnonzero(mask)
        results: Dict[EndKey, object] = {}
        for action in actions:
            next_state = step_fn(s, jnp.int32(action))
            for board_key, end_state in collect(next_state).items():
                results.setdefault(board_key, end_state)
        cache[key] = results
        return results

    return collect(state)


def _sample_dice(rng: random.Random) -> jnp.ndarray:
    idx = rng.choices(range(len(_STOCHASTIC_ACTION_PROBS)), weights=_STOCHASTIC_ACTION_PROBS.tolist(), k=1)[0]
    return _STOCHASTIC_DICE_MAPPING[idx]


def _compare_turn(
    state_bg,
    state_bg2,
    log_lines: List[str],
    rng: random.Random,
) -> Tuple[object, object, bool]:
    t0 = time.perf_counter()
    end_bg = _enumerate_end_states(state_bg, bg_step, _state_key_bg)
    t1 = time.perf_counter()
    end_bg2 = _enumerate_end_states(state_bg2, bg2_step, _state_key_bg2)
    t2 = time.perf_counter()
    bg_ms = (t1 - t0) * 1000.0
    bg2_ms = (t2 - t1) * 1000.0
    log_lines.append(f"End states: bg={len(end_bg)} bg2={len(end_bg2)} timing_ms={bg_ms:.2f}/{bg2_ms:.2f}")
    print(log_lines[-1], flush=True)

    keys_bg = set(end_bg.keys())
    keys_bg2 = set(end_bg2.keys())
    if keys_bg != keys_bg2:
        only_bg = keys_bg - keys_bg2
        only_bg2 = keys_bg2 - keys_bg
        log_lines.append(f"Mismatch: bg_only={len(only_bg)} bg2_only={len(only_bg2)}")
        if only_bg:
            log_lines.append(f"Example bg_only end state key: {next(iter(only_bg))}")
        if only_bg2:
            log_lines.append(f"Example bg2_only end state key: {next(iter(only_bg2))}")
        return state_bg, state_bg2, False

    chosen = rng.choice(list(keys_bg))
    return end_bg[chosen], end_bg2[chosen], True


def _warmup_compare(env_bg, env_bg2, seed: int, rng: random.Random) -> float:
    state_bg = env_bg.init(jax.random.PRNGKey(seed))
    state_bg2 = env_bg2.init(jax.random.PRNGKey(seed))
    if not (bool(state_bg._is_stochastic) and bool(state_bg2._is_stochastic)):
        return 0.0
    dice = _sample_dice(rng)
    state_bg = env_bg.set_dice(state_bg, dice)
    state_bg2 = env_bg2.set_dice(state_bg2, dice)
    t0 = time.perf_counter()
    _enumerate_end_states(state_bg, bg_step, _state_key_bg)
    _enumerate_end_states(state_bg2, bg2_step, _state_key_bg2)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def run_compare(games: int, seed: int, max_turns: int, log_path: str) -> bool:
    rng = random.Random(seed)
    env_bg = Backgammon()
    env_bg2 = Backgammon2P()

    log_lines = []
    warmup_ms = _warmup_compare(env_bg, env_bg2, seed, rng)
    log_lines.append(f"Warmup (compile) time: {warmup_ms:.2f} ms")
    print(log_lines[-1], flush=True)
    ok = True
    for game_idx in range(games):
        state_bg = env_bg.init(jax.random.PRNGKey(seed + game_idx))
        state_bg2 = env_bg2.init(jax.random.PRNGKey(seed + game_idx))
        turns = 0

        while turns < max_turns and not bool(state_bg.terminated) and not bool(state_bg2.terminated):
            if bool(state_bg._is_stochastic) and bool(state_bg2._is_stochastic):
                dice = _sample_dice(rng)
                state_bg = env_bg.set_dice(state_bg, dice)
                state_bg2 = env_bg2.set_dice(state_bg2, dice)
                turns += 1

                log_lines.append(f"Game {game_idx} Turn {turns} Dice {int(dice[0]) + 1}-{int(dice[1]) + 1}")
                print(log_lines[-1], flush=True)
                state_bg, state_bg2, ok = _compare_turn(state_bg, state_bg2, log_lines, rng)
                if not ok:
                    break
            else:
                log_lines.append("State mismatch: one env stochastic and the other not.")
                ok = False
                break

        if ok and turns >= max_turns and (not bool(state_bg.terminated) or not bool(state_bg2.terminated)):
            log_lines.append(
                f"Reached max_turns={max_turns} on game {game_idx} without termination."
            )
            ok = False

        if not ok:
            log_lines.append(f"Stopped on game {game_idx}, turn {turns}.")
            break

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backgammon vs backgammon2p end states.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--log-path", type=str, default="logs/pgx_backgammon2p_diff.log")
    args = parser.parse_args()
    ok = run_compare(args.games, args.seed, args.max_turns, args.log_path)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
