#!/usr/bin/env python3
"""
Benchmark for PGX Backgammon implementation.

This benchmark measures games per second when simulating backgammon games
with random legal action selection, optimized for GPU parallelization.

Usage:
    python benchmarks/benchmark_backgammon.py [--batch-sizes 1,10,100,1000,10000]
"""

import argparse
import json
import os
import subprocess
import time
import warnings
from datetime import datetime
from functools import partial
import statistics
from typing import NamedTuple

# Suppress CUDA interconnect warnings
# Only set default platform if not already specified
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cuda"
warnings.filterwarnings("ignore", message=".*GPU interconnect.*")

import jax
import jax.numpy as jnp

from pgx.backgammon import Backgammon, State as BackgammonState
from pgx.backgammon2p import Backgammon2P

class BenchmarkResult(NamedTuple):
    """Results from a benchmark run."""
    batch_size: int
    num_games: int
    completed_games: int
    timeout_games: int
    total_steps: int
    total_moves: int  # Player moves (non-stochastic steps)
    total_turns: int  # Player turns (current_player changes)
    total_checker_moves: int  # Estimated checker moves executed
    elapsed_time: float
    games_per_second: float
    steps_per_second: float
    moves_per_second: float  # Player moves per second
    turns_per_second: float
    checker_moves_per_second: float
    warmup_time: float
    batch_time_avg: float
    batch_time_min: float
    batch_time_max: float
    batch_time_count: int
    # Game statistics
    avg_steps_per_game: float
    avg_moves_per_game: float
    avg_turns_per_game: float
    avg_checker_moves_per_game: float
    min_steps: int
    max_steps: int
    # Point distribution (1pt, 2pt, 3pt games)
    games_1pt: int
    games_2pt: int
    games_3pt: int


def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_commit_short() -> str:
    """Get the short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_device_info() -> str:
    """Get JAX device information."""
    devices = jax.devices()
    device_strs = [f"{d.platform}:{d.device_kind}" for d in devices]
    return ", ".join(device_strs)


def select_random_action(key: jnp.ndarray, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
    """Select a random action from the legal action mask."""
    logits = jnp.where(
        legal_action_mask,
        0.0,
        -1e9
    )
    return jax.random.categorical(key, logits=logits, axis=-1)


def select_random_stochastic_action(key: jnp.ndarray, num_stochastic_actions: int = 21) -> jnp.ndarray:
    """Select a random stochastic action (dice roll)."""
    return jax.random.randint(key, shape=(), minval=0, maxval=num_stochastic_actions)


def run_single_game(
    env: Backgammon,
    init_fn,
    step_fn,
    stochastic_step_fn,
    key: jnp.ndarray,
    max_steps: int = 10000
) -> tuple[int, bool]:
    """
    Run a single game to completion with random actions.
    Returns (steps_taken, completed).
    """
    key, subkey = jax.random.split(key)
    state = init_fn(subkey)
    steps = 0

    while not state.terminated and steps < max_steps:
        key, subkey1, subkey2 = jax.random.split(key, 3)

        if state._is_stochastic:
            # Roll dice randomly
            action = select_random_stochastic_action(subkey1)
            state = stochastic_step_fn(state, action)
        else:
            # Select random legal action
            action = select_random_action(subkey1, state.legal_action_mask)
            state = step_fn(state, action, subkey2)

        steps += 1

    return steps, bool(state.terminated)


def create_batched_game_loop(env, batch_size: int, max_steps: int = 5000):
    """
    Create JIT-compiled functions for running batched games.

    Returns functions optimized for running many games in parallel using vmap.
    """
    num_stochastic_actions = env.num_stochastic_actions

    init_fn = jax.jit(jax.vmap(env.init))
    det_step_fn = env.step_deterministic
    stochastic_step_fn = env.stochastic_step

    @jax.jit
    def select_actions_batch(
        key: jnp.ndarray, states_legal_mask: jnp.ndarray, states_is_stochastic: jnp.ndarray
    ) -> jnp.ndarray:
        """Select random actions for a batch of states."""
        keys = jax.random.split(key, states_legal_mask.shape[0])

        def choose_action(k, legal_mask, is_stochastic):
            return jax.lax.cond(
                is_stochastic,
                lambda: jax.random.randint(k, shape=(), minval=0, maxval=num_stochastic_actions),
                lambda: jax.random.categorical(k, logits=jnp.where(legal_mask, 0.0, -1e9)),
            )

        return jax.vmap(choose_action)(keys, states_legal_mask, states_is_stochastic)

    @jax.jit
    def step_batch(states, actions: jnp.ndarray):
        """Step a batch of states, dispatching to stochastic or deterministic phase per state."""

        def step_one(state, action):
            # Skip already finished games to preserve terminal rewards/flags
            return jax.lax.cond(
                (state.terminated | state.truncated),
                lambda: state,
                lambda: jax.lax.cond(
                    state._is_stochastic,
                    lambda: stochastic_step_fn(state, action),
                    lambda: det_step_fn(state, action),
                ),
            )

        return jax.vmap(step_one)(states, actions)

    def count_checker_moves(actions: jnp.ndarray, is_stochastic: jnp.ndarray) -> jnp.ndarray:
        """Approximate checker moves per action; stochastic steps count as zero."""
        # backgammon2p reports env.id as "backgammon", so branch on version
        if getattr(env, "version", "") == "2p":
            src1 = actions // 26
            src2 = actions % 26
            move_count = (src1 != 0).astype(jnp.int32) + (src2 != 0).astype(jnp.int32)
        elif env.id == "backgammon":
            src = actions // 6
            move_count = (src != 0).astype(jnp.int32)
        else:
            move_count = jnp.zeros_like(actions, dtype=jnp.int32)
        return jnp.where(is_stochastic, 0, move_count)

    def game_step(carry):
        """Single step of the game loop - runs entirely in JAX."""
        states, steps_taken, moves_taken, turns_taken, checker_moves_taken, key, step_count = carry

        # Generate keys for this step
        key, action_key = jax.random.split(key, 2)

        # Select actions for all games
        actions = select_actions_batch(action_key, states.legal_action_mask, states._is_stochastic)

        # Step all games
        next_states = step_batch(states, actions)

        # Checker moves for deterministic actions
        checker_moves = count_checker_moves(actions, states._is_stochastic)

        # Update step counts for running games (before this step)
        running = ~states.terminated
        steps_taken = steps_taken + running.astype(jnp.int32)

        # Update move counts (only for non-stochastic steps = player moves)
        is_move = running & ~states._is_stochastic
        moves_taken = moves_taken + is_move.astype(jnp.int32)

        # Track turns (current_player change on still-running games)
        turn_change = running & (next_states.current_player != states.current_player)
        turns_taken = turns_taken + turn_change.astype(jnp.int32)

        # Checker moves
        checker_moves_taken = checker_moves_taken + checker_moves * running.astype(jnp.int32)

        return (next_states, steps_taken, moves_taken, turns_taken, checker_moves_taken, key, step_count + 1)

    def continue_condition(carry):
        """Continue while any game is still running and under max_steps."""
        states, steps_taken, moves_taken, turns_taken, checker_moves_taken, key, step_count = carry
        any_running = jnp.any(~states.terminated)
        under_limit = step_count < max_steps
        return any_running & under_limit

    @jax.jit
    def run_batched_games(key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run a batch of games to completion using JAX while_loop.
        Returns (steps_per_game, moves_per_game, turns_per_game, checker_moves_per_game, completed_mask, win_scores).
        """
        # Initialize batch of games
        keys = jax.random.split(key, batch_size + 1)
        key = keys[0]
        init_keys = keys[1:]
        states = init_fn(init_keys)

        steps_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        moves_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        turns_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        checker_moves_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        step_count = jnp.int32(0)

        # Run the game loop entirely in JAX
        initial_carry = (states, steps_taken, moves_taken, turns_taken, checker_moves_taken, key, step_count)
        final_states, final_steps, final_moves, final_turns, final_checker_moves, _, _ = jax.lax.while_loop(
            continue_condition,
            game_step,
            initial_carry
        )

        # Get win scores (absolute value of max reward per game)
        win_scores = jnp.abs(final_states.rewards).max(axis=-1)
        return final_steps, final_moves, final_turns, final_checker_moves, final_states.terminated, win_scores

    return run_batched_games, init_fn


def _make_env(env_name: str, short_game: bool):
    if env_name == "backgammon":
        return Backgammon(short_game=short_game)
    if env_name == "backgammon2p":
        return Backgammon2P(short_game=short_game)
    raise ValueError(f"Unknown env: {env_name}")


def run_benchmark(
    env_name: str,
    batch_size: int,
    num_batches: int = 10,
    max_steps: int = 5000,
    warmup_batches: int = 2,
    short_game: bool = False,
    profile: bool = False
) -> BenchmarkResult:
    """
    Run the benchmark with the specified batch size.

    Args:
        env_name: Name of the env to benchmark ("backgammon", "backgammon2p")
        batch_size: Number of games to run in parallel
        num_batches: Number of batches to run for timing
        max_steps: Maximum steps per game before timeout
        warmup_batches: Number of warmup batches (not timed)
        short_game: Use shorter game variant for faster testing

    Returns:
        BenchmarkResult with timing statistics
    """
    env = _make_env(env_name, short_game=short_game)
    run_batched_games, init_fn = create_batched_game_loop(env, batch_size, max_steps)

    key = jax.random.PRNGKey(42)

    # Warmup runs (for JIT compilation)
    print(f"  Warming up ({warmup_batches} batches)...", end="", flush=True)
    warmup_start = time.perf_counter()
    for i in range(warmup_batches):
        key, subkey = jax.random.split(key)
        steps, moves, turns, checker_moves, completed, win_scores = run_batched_games(subkey)
        jax.block_until_ready(steps)
    warmup_elapsed = time.perf_counter() - warmup_start
    print(" done")
    if profile:
        print(f"    Warmup time: {warmup_elapsed:.2f}s")

    # Timed runs
    print(f"  Running {num_batches} batches...", end="", flush=True)
    total_games = 0
    total_steps = 0
    total_moves = 0
    total_turns = 0
    total_checker_moves = 0
    total_completed = 0
    total_timeouts = 0
    all_steps = []
    all_moves = []
    all_turns = []
    all_checker_moves = []
    all_win_scores = []
    all_completed = []
    batch_times = []

    start_time = time.perf_counter()

    for i in range(num_batches):
        key, subkey = jax.random.split(key)
        batch_start = time.perf_counter()
        steps, moves, turns, checker_moves, completed, win_scores = run_batched_games(subkey)
        jax.block_until_ready(steps)
        batch_times.append(time.perf_counter() - batch_start)
        total_steps += int(jnp.sum(steps))
        total_moves += int(jnp.sum(moves))
        total_turns += int(jnp.sum(turns))
        total_checker_moves += int(jnp.sum(checker_moves))
        total_games += batch_size
        completed_count = int(jnp.sum(completed))
        total_completed += completed_count
        total_timeouts += batch_size - completed_count
        all_steps.append(steps)
        all_moves.append(moves)
        all_turns.append(turns)
        all_checker_moves.append(checker_moves)
        all_win_scores.append(win_scores)
        all_completed.append(completed)

    elapsed_time = time.perf_counter() - start_time
    print(" done")
    if profile and batch_times:
        print(
            f"    Batch time (s): avg={statistics.mean(batch_times):.3f} "
            f"min={min(batch_times):.3f} max={max(batch_times):.3f}"
        )
    batch_time_avg = statistics.mean(batch_times) if batch_times else 0.0
    batch_time_min = min(batch_times) if batch_times else 0.0
    batch_time_max = max(batch_times) if batch_times else 0.0
    batch_time_count = len(batch_times)

    # Aggregate all steps, moves, and win scores
    all_steps = jnp.concatenate(all_steps)
    all_moves = jnp.concatenate(all_moves)
    all_turns = jnp.concatenate(all_turns)
    all_checker_moves = jnp.concatenate(all_checker_moves)
    all_win_scores = jnp.concatenate(all_win_scores)
    all_completed = jnp.concatenate(all_completed)

    games_per_second = total_games / elapsed_time
    steps_per_second = total_steps / elapsed_time
    moves_per_second = total_moves / elapsed_time
    turns_per_second = total_turns / elapsed_time if elapsed_time > 0 else 0.0
    checker_moves_per_second = total_checker_moves / elapsed_time if elapsed_time > 0 else 0.0

    # Calculate game statistics
    avg_steps_per_game = float(jnp.mean(all_steps))
    avg_moves_per_game = float(jnp.mean(all_moves))
    avg_turns_per_game = float(jnp.mean(all_turns))
    avg_checker_moves_per_game = float(jnp.mean(all_checker_moves))
    min_steps = int(jnp.min(all_steps))
    max_steps = int(jnp.max(all_steps))

    # Count point distribution (only completed games)
    if total_completed > 0:
        completed_win_scores = all_win_scores[all_completed]
        games_1pt = int(jnp.sum(completed_win_scores == 1))
        games_2pt = int(jnp.sum(completed_win_scores == 2))
        games_3pt = int(jnp.sum(completed_win_scores == 3))
    else:
        games_1pt = games_2pt = games_3pt = 0

    return BenchmarkResult(
        batch_size=batch_size,
        num_games=total_games,
        completed_games=total_completed,
        timeout_games=total_timeouts,
        total_steps=total_steps,
        total_moves=total_moves,
        total_turns=total_turns,
        total_checker_moves=total_checker_moves,
        elapsed_time=elapsed_time,
        games_per_second=games_per_second,
        steps_per_second=steps_per_second,
        moves_per_second=moves_per_second,
        turns_per_second=turns_per_second,
        checker_moves_per_second=checker_moves_per_second,
        warmup_time=warmup_elapsed,
        batch_time_avg=batch_time_avg,
        batch_time_min=batch_time_min,
        batch_time_max=batch_time_max,
        batch_time_count=batch_time_count,
        avg_steps_per_game=avg_steps_per_game,
        avg_moves_per_game=avg_moves_per_game,
        avg_turns_per_game=avg_turns_per_game,
        avg_checker_moves_per_game=avg_checker_moves_per_game,
        min_steps=min_steps,
        max_steps=max_steps,
        games_1pt=games_1pt,
        games_2pt=games_2pt,
        games_3pt=games_3pt
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark PGX Backgammon")
    parser.add_argument(
        "--envs",
        type=str,
        default="backgammon",
        help="Comma-separated envs to benchmark (backgammon,backgammon2p)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,10,100,1000,10000",
        help="Comma-separated list of batch sizes to test"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to run per batch size"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum steps per game"
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Number of warmup batches"
    )
    parser.add_argument(
        "--short-game",
        action="store_true",
        help="Use short game variant (faster games)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to JSON file for saving results (appends to existing file)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark mode: single batch size (1000), 3 batches, short game"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print warmup and per-batch timing details"
    )
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        batch_sizes = [1000]
        args.num_batches = 3
        args.short_game = True
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]

    # Print header
    print("=" * 70)
    print("PGX Backgammon Benchmark")
    print("=" * 70)
    print(f"Git Commit: {get_git_commit()}")
    print(f"Git Commit (short): {get_git_commit_short()}")
    print(f"Device(s): {get_device_info()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Envs: {envs}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Batches per size: {args.num_batches}")
    print(f"Max steps per game: {args.max_steps}")
    print(f"Short game mode: {args.short_game}")
    print("=" * 70)
    print()

    all_results = {}

    for env_name in envs:
        print(f"\n===== ENV: {env_name} =====")
        results = []

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            print("-" * 40)

            result = run_benchmark(
                env_name=env_name,
                batch_size=batch_size,
                num_batches=args.num_batches,
                max_steps=args.max_steps,
                warmup_batches=args.warmup_batches,
                short_game=args.short_game,
                profile=args.profile
            )
            results.append(result)

            print(f"  Total games: {result.num_games:,}")
            print(f"  Completed: {result.completed_games:,} ({(result.completed_games / result.num_games * 100):.1f}%) | Timeouts: {result.timeout_games:,}")
            print(f"  Total steps: {result.total_steps:,}")
            print(f"  Total moves: {result.total_moves:,}")
            print(f"  Total turns: {result.total_turns:,}")
            print(f"  Total checker moves: {result.total_checker_moves:,}")
            print(f"  Elapsed time: {result.elapsed_time:.2f}s")
            print(f"  Games/second: {result.games_per_second:,.1f}")
            print(f"  Steps/second: {result.steps_per_second:,.1f}")
            print(f"  Moves/second: {result.moves_per_second:,.1f}")
            print(f"  Turns/second: {result.turns_per_second:,.1f}")
            print(f"  Checker moves/second: {result.checker_moves_per_second:,.1f}")
            if args.profile:
                print(
                    f"  Warmup: {result.warmup_time:.2f}s | "
                    f"Batch time (s) avg/min/max: {result.batch_time_avg:.3f}/"
                    f"{result.batch_time_min:.3f}/{result.batch_time_max:.3f} "
                    f"({result.batch_time_count} batches)"
                )
            print(
                f"  Avg steps/game: {result.avg_steps_per_game:.1f} "
                f"(moves={result.avg_moves_per_game:.1f}, turns={result.avg_turns_per_game:.1f}, "
                f"checker_moves={result.avg_checker_moves_per_game:.1f})"
            )
            print(f"  Min/Max steps: {result.min_steps} / {result.max_steps}")
            print(f"  Points: 1pt={result.games_1pt}, 2pt={result.games_2pt}, 3pt={result.games_3pt}")

        print("\n" + "=" * 70)
        print(f"SUMMARY ({env_name})")
        print("=" * 70)
        print(f"{'Batch Size':>12} {'Games/sec':>12} {'Steps/sec':>12} {'Moves/sec':>12} {'Time (s)':>10}")
        print("-" * 60)
        for r in results:
            print(f"{r.batch_size:>12,} {r.games_per_second:>12,.1f} {r.steps_per_second:>12,.1f} {r.moves_per_second:>12,.1f} {r.elapsed_time:>10.2f}")

        best = max(results, key=lambda r: r.games_per_second)
        print("-" * 60)
        print(f"Best throughput: {best.games_per_second:,.1f} games/sec at batch size {best.batch_size:,}")
        print("=" * 70)

        all_results[env_name] = results

    if args.output_json:
        for env_name, results in all_results.items():
            save_results_to_json(
                env_name=env_name,
                results=results,
                output_path=args.output_json,
                git_commit=get_git_commit(),
                git_commit_short=get_git_commit_short(),
                device_info=get_device_info(),
                jax_version=jax.__version__,
                short_game=args.short_game,
                max_steps=args.max_steps,
                num_batches=args.num_batches
            )


def save_results_to_json(
    env_name: str,
    results: list[BenchmarkResult],
    output_path: str,
    git_commit: str,
    git_commit_short: str,
    device_info: str,
    jax_version: str,
    short_game: bool,
    max_steps: int,
    num_batches: int
):
    """Save benchmark results to a JSON file, appending to existing results."""
    # Find best result
    best = max(results, key=lambda r: r.games_per_second)

    # Create the new entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "git_commit_short": git_commit_short,
        "device": device_info,
        "jax_version": jax_version,
        "env": env_name,
        "config": {
            "short_game": short_game,
            "max_steps": max_steps,
            "num_batches": num_batches
        },
        "best_result": {
            "batch_size": best.batch_size,
            "games_per_second": best.games_per_second,
            "steps_per_second": best.steps_per_second,
            "moves_per_second": best.moves_per_second,
            "turns_per_second": best.turns_per_second,
            "checker_moves_per_second": best.checker_moves_per_second,
        },
        "all_results": [
            {
                "batch_size": r.batch_size,
                "num_games": r.num_games,
                "completed_games": r.completed_games,
                "timeout_games": r.timeout_games,
                "total_steps": r.total_steps,
                "total_moves": r.total_moves,
                "total_turns": r.total_turns,
                "total_checker_moves": r.total_checker_moves,
                "elapsed_time": r.elapsed_time,
                "games_per_second": r.games_per_second,
                "steps_per_second": r.steps_per_second,
                "moves_per_second": r.moves_per_second,
                "turns_per_second": r.turns_per_second,
                "checker_moves_per_second": r.checker_moves_per_second,
                "warmup_time": r.warmup_time,
                "batch_time_avg": r.batch_time_avg,
                "batch_time_min": r.batch_time_min,
                "batch_time_max": r.batch_time_max,
                "batch_time_count": r.batch_time_count,
                "avg_steps_per_game": r.avg_steps_per_game,
                "avg_moves_per_game": r.avg_moves_per_game,
                "avg_turns_per_game": r.avg_turns_per_game,
                "avg_checker_moves_per_game": r.avg_checker_moves_per_game,
                "min_steps": r.min_steps,
                "max_steps": r.max_steps,
                "games_1pt": r.games_1pt,
                "games_2pt": r.games_2pt,
                "games_3pt": r.games_3pt
            }
            for r in results
        ]
    }

    # Load existing data or create new
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        data = {"benchmark_runs": []}

    # Append new entry
    data["benchmark_runs"].append(new_entry)

    # Save back
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
