#!/usr/bin/env python3
"""
Benchmark for PGX 2048 implementation.

This benchmark measures games per second when simulating 2048 games
with random legal action selection, optimized for GPU parallelization.

Usage:
    python benchmarks/benchmark_2048.py [--batch-sizes 1,10,100,1000,10000]
"""

import argparse
import json
import os
import subprocess
import time
import warnings
from datetime import datetime
from typing import NamedTuple

# Suppress CUDA interconnect warnings
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cuda"
warnings.filterwarnings("ignore", message=".*GPU interconnect.*")

import jax
import jax.numpy as jnp

from pgx.play2048 import Play2048, State


class BenchmarkResult(NamedTuple):
    """Results from a benchmark run."""
    batch_size: int
    num_games: int
    total_steps: int
    elapsed_time: float
    games_per_second: float
    steps_per_second: float
    avg_steps_per_game: float
    min_steps: int
    max_steps: int
    avg_score: float
    max_score: float


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


def create_batched_game_loop(env: Play2048, batch_size: int, max_steps: int = 5000):
    """
    Create JIT-compiled functions for running batched games.
    """
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    @jax.jit
    def select_actions_batch(key: jnp.ndarray, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
        """Select random actions for a batch of states."""
        keys = jax.random.split(key, legal_action_mask.shape[0])
        logits = jnp.where(legal_action_mask, 0.0, -1e9)
        return jax.vmap(
            lambda k, l: jax.random.categorical(k, logits=l)
        )(keys, logits)

    def game_step(carry):
        """Single step of the game loop."""
        states, steps_taken, total_reward, key, step_count = carry

        key, action_key, step_key = jax.random.split(key, 3)
        step_keys = jax.random.split(step_key, batch_size)

        actions = select_actions_batch(action_key, states.legal_action_mask)
        next_states = jax.vmap(env.step)(states, actions, step_keys)

        running = ~states.terminated
        steps_taken = steps_taken + running.astype(jnp.int32)
        total_reward = total_reward + jnp.where(running, next_states.rewards[:, 0], 0.0)

        return (next_states, steps_taken, total_reward, key, step_count + 1)

    def continue_condition(carry):
        """Continue while any game is still running and under max_steps."""
        states, steps_taken, total_reward, key, step_count = carry
        any_running = jnp.any(~states.terminated)
        under_limit = step_count < max_steps
        return any_running & under_limit

    @jax.jit
    def run_batched_games(key: jnp.ndarray) -> tuple:
        """Run a batch of games to completion."""
        keys = jax.random.split(key, batch_size + 1)
        key = keys[0]
        init_keys = keys[1:]
        states = init_fn(init_keys)

        steps_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        total_reward = jnp.zeros(batch_size, dtype=jnp.float32)
        step_count = jnp.int32(0)

        initial_carry = (states, steps_taken, total_reward, key, step_count)
        final_states, final_steps, final_rewards, _, _ = jax.lax.while_loop(
            continue_condition,
            game_step,
            initial_carry
        )

        return final_steps, final_states.terminated, final_rewards

    return run_batched_games, init_fn


def run_benchmark(
    batch_size: int,
    num_batches: int = 10,
    max_steps: int = 5000,
    warmup_batches: int = 2
) -> BenchmarkResult:
    """Run the benchmark with the specified batch size."""
    env = Play2048()
    run_batched_games, init_fn = create_batched_game_loop(env, batch_size, max_steps)

    key = jax.random.PRNGKey(42)

    # Warmup runs
    print(f"  Warming up ({warmup_batches} batches)...", end="", flush=True)
    for i in range(warmup_batches):
        key, subkey = jax.random.split(key)
        steps, completed, scores = run_batched_games(subkey)
        jax.block_until_ready(steps)
    print(" done")

    # Timed runs
    print(f"  Running {num_batches} batches...", end="", flush=True)
    total_games = 0
    total_steps = 0
    all_steps = []
    all_scores = []

    start_time = time.perf_counter()

    for i in range(num_batches):
        key, subkey = jax.random.split(key)
        steps, completed, scores = run_batched_games(subkey)
        jax.block_until_ready(steps)
        total_steps += int(jnp.sum(steps))
        total_games += batch_size
        all_steps.append(steps)
        all_scores.append(scores)

    elapsed_time = time.perf_counter() - start_time
    print(" done")

    all_steps = jnp.concatenate(all_steps)
    all_scores = jnp.concatenate(all_scores)

    games_per_second = total_games / elapsed_time
    steps_per_second = total_steps / elapsed_time

    return BenchmarkResult(
        batch_size=batch_size,
        num_games=total_games,
        total_steps=total_steps,
        elapsed_time=elapsed_time,
        games_per_second=games_per_second,
        steps_per_second=steps_per_second,
        avg_steps_per_game=float(jnp.mean(all_steps)),
        min_steps=int(jnp.min(all_steps)),
        max_steps=int(jnp.max(all_steps)),
        avg_score=float(jnp.mean(all_scores)),
        max_score=float(jnp.max(all_scores))
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark PGX 2048")
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
        "--output-json",
        type=str,
        default=None,
        help="Path to JSON file for saving results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark mode: batch size 1000, 3 batches"
    )
    args = parser.parse_args()

    if args.quick:
        batch_sizes = [1000]
        args.num_batches = 3
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("=" * 70)
    print("PGX 2048 Benchmark")
    print("=" * 70)
    print(f"Git Commit: {get_git_commit_short()}")
    print(f"Device(s): {get_device_info()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Batches per size: {args.num_batches}")
    print(f"Max steps per game: {args.max_steps}")
    print("=" * 70)
    print()

    results = []

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 40)

        result = run_benchmark(
            batch_size=batch_size,
            num_batches=args.num_batches,
            max_steps=args.max_steps,
            warmup_batches=args.warmup_batches
        )
        results.append(result)

        print(f"  Total games: {result.num_games:,}")
        print(f"  Total steps: {result.total_steps:,}")
        print(f"  Elapsed time: {result.elapsed_time:.2f}s")
        print(f"  Games/second: {result.games_per_second:,.1f}")
        print(f"  Steps/second: {result.steps_per_second:,.1f}")
        print(f"  Avg steps/game: {result.avg_steps_per_game:.1f}")
        print(f"  Min/Max steps: {result.min_steps} / {result.max_steps}")
        print(f"  Avg/Max score: {result.avg_score:,.0f} / {result.max_score:,.0f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch Size':>12} {'Games/sec':>12} {'Steps/sec':>14} {'Time (s)':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r.batch_size:>12,} {r.games_per_second:>12,.1f} {r.steps_per_second:>14,.1f} {r.elapsed_time:>10.2f}")

    best = max(results, key=lambda r: r.games_per_second)
    print("-" * 50)
    print(f"Best throughput: {best.games_per_second:,.1f} games/sec at batch size {best.batch_size:,}")
    print("=" * 70)

    if args.output_json:
        save_results_to_json(results, args.output_json, get_git_commit_short(), get_device_info())


def save_results_to_json(results, output_path, git_commit, device_info):
    """Save benchmark results to JSON."""
    best = max(results, key=lambda r: r.games_per_second)

    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "device": device_info,
        "best_result": {
            "batch_size": best.batch_size,
            "games_per_second": best.games_per_second,
            "steps_per_second": best.steps_per_second
        },
        "all_results": [
            {
                "batch_size": r.batch_size,
                "num_games": r.num_games,
                "total_steps": r.total_steps,
                "elapsed_time": r.elapsed_time,
                "games_per_second": r.games_per_second,
                "steps_per_second": r.steps_per_second,
                "avg_steps_per_game": r.avg_steps_per_game,
                "avg_score": r.avg_score,
                "max_score": r.max_score
            }
            for r in results
        ]
    }

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        data = {"benchmark_runs": []}

    data["benchmark_runs"].append(new_entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
