#!/usr/bin/env python3
"""
Comprehensive benchmark for all game variants.

This script benchmarks all optimized variants of 2048 and Backgammon,
validates correctness, and outputs performance comparisons.

Usage:
    # Quick benchmark (CPU validation + performance test)
    python benchmarks/benchmark_all_variants.py --quick

    # Full benchmark with more batches
    python benchmarks/benchmark_all_variants.py --batch-size 1000 --num-batches 5

    # Correctness validation only
    python benchmarks/benchmark_all_variants.py --validate-only

    # GPU/TPU benchmark (for Google Colab)
    python benchmarks/benchmark_all_variants.py --batch-size 4000 --num-batches 10
"""

import argparse
import json
import os
import subprocess
import time
import warnings
from datetime import datetime
from typing import NamedTuple, Dict, List, Any

# Suppress CUDA warnings for cleaner output
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"  # Default to CPU for validation
warnings.filterwarnings("ignore", message=".*GPU interconnect.*")

import jax
import jax.numpy as jnp


class BenchmarkResult(NamedTuple):
    """Results from a benchmark run."""
    variant_name: str
    game: str
    batch_size: int
    num_games: int
    total_steps: int
    elapsed_time: float
    games_per_second: float
    steps_per_second: float


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


# =============================================================================
# 2048 Variants
# =============================================================================

def get_2048_variants():
    """Import and return all 2048 variants."""
    from pgx.play2048 import Play2048, State as State2048Orig
    from pgx.play2048_v2_branchless import Play2048V2Branchless, State as State2048Branchless
    from pgx.play2048_v2_no_rotate import Play2048V2NoRotate, State as State2048NoRotate
    from pgx.play2048_v2_all import Play2048V2All, State as State2048All

    return {
        "original": (Play2048, State2048Orig),
        "branchless": (Play2048V2Branchless, State2048Branchless),
        "no_rotate": (Play2048V2NoRotate, State2048NoRotate),
        "all": (Play2048V2All, State2048All),
    }


# =============================================================================
# Backgammon Variants
# =============================================================================

def get_backgammon_variants():
    """Import and return all Backgammon variants."""
    from pgx.backgammon import Backgammon, State as StateBGOrig
    from pgx.backgammon_v2_fast_obs import BackgammonV2FastObs, State as StateBGFastObs
    from pgx.backgammon_v2_branchless import BackgammonV2Branchless, State as StateBGBranchless
    from pgx.backgammon_v2_all import BackgammonV2All, State as StateBGAll

    return {
        "original": (Backgammon, StateBGOrig),
        "fast_obs": (BackgammonV2FastObs, StateBGFastObs),
        "branchless": (BackgammonV2Branchless, StateBGBranchless),
        "all": (BackgammonV2All, StateBGAll),
    }


# =============================================================================
# Correctness Validation
# =============================================================================

def validate_2048_variant(variant_name: str, EnvClass, StateClass, num_games: int = 100):
    """
    Validate a 2048 variant by comparing game outcomes with the original.

    Returns True if the variant produces correct game mechanics.
    """
    from pgx.play2048 import Play2048

    print(f"  Validating 2048 {variant_name}...", end=" ", flush=True)

    env_orig = Play2048()
    env_test = EnvClass()

    errors = []

    for seed in range(num_games):
        key = jax.random.PRNGKey(seed)

        # Initialize both
        key, k1, k2 = jax.random.split(key, 3)
        state_orig = env_orig.init(k1)
        state_test = env_test.init(k1)  # Same key for same init

        # Compare initial boards
        if not jnp.allclose(state_orig._board, state_test._board):
            errors.append(f"Game {seed}: Initial board mismatch")
            continue

        # Run a few steps with deterministic actions
        max_steps = 50
        for step in range(max_steps):
            if state_orig.terminated or state_test.terminated:
                break

            # Get legal actions
            legal_orig = state_orig.legal_action_mask
            legal_test = state_test.legal_action_mask

            if not jnp.allclose(legal_orig, legal_test):
                errors.append(f"Game {seed}, Step {step}: Legal action mask mismatch")
                break

            # Select same action for both
            key, subkey = jax.random.split(key)
            logits = jnp.where(legal_orig, 0.0, -1e9)
            action = jax.random.categorical(subkey, logits=logits)

            # Step both environments
            key, k1 = jax.random.split(key)
            state_orig = env_orig.step(state_orig, action, k1)
            state_test = env_test.step(state_test, action, k1)

            # Compare boards after step
            if not jnp.allclose(state_orig._board, state_test._board):
                errors.append(f"Game {seed}, Step {step}: Board mismatch after action {action}")
                break

    if errors:
        print(f"FAILED ({len(errors)} errors)")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    else:
        print("PASSED")
        return True


def validate_backgammon_variant(variant_name: str, EnvClass, StateClass, num_games: int = 50):
    """
    Validate a Backgammon variant by comparing game mechanics with the original.

    Note: Observation size may differ between variants, so we only compare
    board state and legal action masks.
    """
    from pgx.backgammon import Backgammon

    print(f"  Validating Backgammon {variant_name}...", end=" ", flush=True)

    env_orig = Backgammon(short_game=True)
    env_test = EnvClass(short_game=True)

    errors = []

    for seed in range(num_games):
        key = jax.random.PRNGKey(seed)

        # Initialize both
        key, k1 = jax.random.split(key)
        state_orig = env_orig.init(k1)
        state_test = env_test.init(k1)

        # Compare initial boards
        if not jnp.allclose(state_orig._board, state_test._board):
            errors.append(f"Game {seed}: Initial board mismatch")
            continue

        # Run a few steps
        max_steps = 100
        for step in range(max_steps):
            if state_orig.terminated or state_test.terminated:
                # Check termination matches
                if state_orig.terminated != state_test.terminated:
                    errors.append(f"Game {seed}, Step {step}: Termination mismatch")
                break

            # Handle stochastic vs deterministic steps
            if state_orig._is_stochastic:
                # Roll dice - use same roll for both
                key, subkey = jax.random.split(key)
                dice_action = jax.random.randint(subkey, shape=(), minval=0, maxval=21)

                state_orig = env_orig.step_stochastic(state_orig, dice_action)
                state_test = env_test.step_stochastic(state_test, dice_action)
            else:
                # Compare legal action masks
                legal_orig = state_orig.legal_action_mask
                legal_test = state_test.legal_action_mask

                if not jnp.allclose(legal_orig, legal_test):
                    errors.append(f"Game {seed}, Step {step}: Legal action mask mismatch")
                    break

                # Select same action for both
                key, subkey = jax.random.split(key)
                logits = jnp.where(legal_orig, 0.0, -1e9)
                action = jax.random.categorical(subkey, logits=logits)

                # Step both
                key, k1 = jax.random.split(key)
                state_orig = env_orig.step(state_orig, action, k1)
                state_test = env_test.step(state_test, action, k1)

            # Compare boards
            if not jnp.allclose(state_orig._board, state_test._board):
                errors.append(f"Game {seed}, Step {step}: Board mismatch")
                break

    if errors:
        print(f"FAILED ({len(errors)} errors)")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    else:
        print("PASSED")
        return True


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_2048_variant(
    variant_name: str,
    EnvClass,
    StateClass,
    batch_size: int,
    num_batches: int,
    max_steps: int = 5000,
    warmup_batches: int = 2
) -> BenchmarkResult:
    """Benchmark a 2048 variant."""
    env = EnvClass()

    init_fn = jax.jit(jax.vmap(env.init))

    @jax.jit
    def select_actions_batch(key, legal_action_mask):
        keys = jax.random.split(key, legal_action_mask.shape[0])
        logits = jnp.where(legal_action_mask, 0.0, -1e9)
        return jax.vmap(lambda k, l: jax.random.categorical(k, logits=l))(keys, logits)

    step_fn = jax.jit(jax.vmap(env.step))

    def game_step(carry):
        states, steps_taken, key, step_count = carry
        key, action_key, step_key = jax.random.split(key, 3)
        step_keys = jax.random.split(step_key, batch_size)
        actions = select_actions_batch(action_key, states.legal_action_mask)
        next_states = step_fn(states, actions, step_keys)
        running = ~states.terminated
        steps_taken = steps_taken + running.astype(jnp.int32)
        return (next_states, steps_taken, key, step_count + 1)

    def continue_condition(carry):
        states, steps_taken, key, step_count = carry
        any_running = jnp.any(~states.terminated)
        under_limit = step_count < max_steps
        return any_running & under_limit

    @jax.jit
    def run_batched_games(key):
        keys = jax.random.split(key, batch_size + 1)
        key = keys[0]
        init_keys = keys[1:]
        states = init_fn(init_keys)
        steps_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        step_count = jnp.int32(0)
        initial_carry = (states, steps_taken, key, step_count)
        final_states, final_steps, _, _ = jax.lax.while_loop(
            continue_condition, game_step, initial_carry
        )
        return final_steps, final_states.terminated

    key = jax.random.PRNGKey(42)

    # Warmup
    for _ in range(warmup_batches):
        key, subkey = jax.random.split(key)
        steps, completed = run_batched_games(subkey)
        jax.block_until_ready(steps)

    # Timed runs
    total_games = 0
    total_steps = 0

    start_time = time.perf_counter()
    for _ in range(num_batches):
        key, subkey = jax.random.split(key)
        steps, completed = run_batched_games(subkey)
        jax.block_until_ready(steps)
        total_steps += int(jnp.sum(steps))
        total_games += batch_size
    elapsed_time = time.perf_counter() - start_time

    return BenchmarkResult(
        variant_name=variant_name,
        game="2048",
        batch_size=batch_size,
        num_games=total_games,
        total_steps=total_steps,
        elapsed_time=elapsed_time,
        games_per_second=total_games / elapsed_time,
        steps_per_second=total_steps / elapsed_time,
    )


def benchmark_backgammon_variant(
    variant_name: str,
    EnvClass,
    StateClass,
    batch_size: int,
    num_batches: int,
    max_steps: int = 5000,
    warmup_batches: int = 2,
    short_game: bool = True
) -> BenchmarkResult:
    """Benchmark a Backgammon variant."""
    env = EnvClass(short_game=short_game)

    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))
    stochastic_step_fn = jax.jit(jax.vmap(env.step_stochastic))

    @jax.jit
    def select_actions_batch(key, legal_mask, is_stochastic):
        keys = jax.random.split(key, legal_mask.shape[0])
        regular_logits = jnp.where(legal_mask, 0.0, -1e9)
        regular_actions = jax.vmap(lambda k, l: jax.random.categorical(k, logits=l))(keys, regular_logits)
        stochastic_actions = jax.vmap(lambda k: jax.random.randint(k, shape=(), minval=0, maxval=21))(keys)
        return jnp.where(is_stochastic, stochastic_actions, regular_actions)

    @jax.jit
    def step_batch(states, actions, keys):
        is_stochastic = states._is_stochastic
        was_terminated = states.terminated
        regular_next = step_fn(states, actions, keys)
        stochastic_next = stochastic_step_fn(states, actions)

        def select_field_with_terminated(orig, reg, stoch, stoch_mask, term_mask):
            while stoch_mask.ndim < reg.ndim:
                stoch_mask = stoch_mask[..., jnp.newaxis]
            next_val = jnp.where(stoch_mask, stoch, reg)
            while term_mask.ndim < orig.ndim:
                term_mask = term_mask[..., jnp.newaxis]
            return jnp.where(term_mask, orig, next_val)

        return StateClass(
            current_player=select_field_with_terminated(states.current_player, regular_next.current_player, stochastic_next.current_player, is_stochastic, was_terminated),
            observation=select_field_with_terminated(states.observation, regular_next.observation, stochastic_next.observation, is_stochastic, was_terminated),
            rewards=select_field_with_terminated(states.rewards, regular_next.rewards, stochastic_next.rewards, is_stochastic, was_terminated),
            terminated=select_field_with_terminated(states.terminated, regular_next.terminated, stochastic_next.terminated, is_stochastic, was_terminated),
            truncated=select_field_with_terminated(states.truncated, regular_next.truncated, stochastic_next.truncated, is_stochastic, was_terminated),
            _is_stochastic=select_field_with_terminated(states._is_stochastic, regular_next._is_stochastic, stochastic_next._is_stochastic, is_stochastic, was_terminated),
            legal_action_mask=select_field_with_terminated(states.legal_action_mask, regular_next.legal_action_mask, stochastic_next.legal_action_mask, is_stochastic, was_terminated),
            _step_count=select_field_with_terminated(states._step_count, regular_next._step_count, stochastic_next._step_count, is_stochastic, was_terminated),
            _board=select_field_with_terminated(states._board, regular_next._board, stochastic_next._board, is_stochastic, was_terminated),
            _dice=select_field_with_terminated(states._dice, regular_next._dice, stochastic_next._dice, is_stochastic, was_terminated),
            _playable_dice=select_field_with_terminated(states._playable_dice, regular_next._playable_dice, stochastic_next._playable_dice, is_stochastic, was_terminated),
            _played_dice_num=select_field_with_terminated(states._played_dice_num, regular_next._played_dice_num, stochastic_next._played_dice_num, is_stochastic, was_terminated),
            _turn=select_field_with_terminated(states._turn, regular_next._turn, stochastic_next._turn, is_stochastic, was_terminated),
        )

    def game_step(carry):
        states, steps_taken, key, step_count = carry
        key, action_key, step_key = jax.random.split(key, 3)
        step_keys = jax.random.split(step_key, batch_size)
        actions = select_actions_batch(action_key, states.legal_action_mask, states._is_stochastic)
        next_states = step_batch(states, actions, step_keys)
        running = ~states.terminated
        steps_taken = steps_taken + running.astype(jnp.int32)
        return (next_states, steps_taken, key, step_count + 1)

    def continue_condition(carry):
        states, steps_taken, key, step_count = carry
        any_running = jnp.any(~states.terminated)
        under_limit = step_count < max_steps
        return any_running & under_limit

    @jax.jit
    def run_batched_games(key):
        keys = jax.random.split(key, batch_size + 1)
        key = keys[0]
        init_keys = keys[1:]
        states = init_fn(init_keys)
        steps_taken = jnp.zeros(batch_size, dtype=jnp.int32)
        step_count = jnp.int32(0)
        initial_carry = (states, steps_taken, key, step_count)
        final_states, final_steps, _, _ = jax.lax.while_loop(
            continue_condition, game_step, initial_carry
        )
        return final_steps, final_states.terminated

    key = jax.random.PRNGKey(42)

    # Warmup
    for _ in range(warmup_batches):
        key, subkey = jax.random.split(key)
        steps, completed = run_batched_games(subkey)
        jax.block_until_ready(steps)

    # Timed runs
    total_games = 0
    total_steps = 0

    start_time = time.perf_counter()
    for _ in range(num_batches):
        key, subkey = jax.random.split(key)
        steps, completed = run_batched_games(subkey)
        jax.block_until_ready(steps)
        total_steps += int(jnp.sum(steps))
        total_games += batch_size
    elapsed_time = time.perf_counter() - start_time

    return BenchmarkResult(
        variant_name=variant_name,
        game="backgammon",
        batch_size=batch_size,
        num_games=total_games,
        total_steps=total_steps,
        elapsed_time=elapsed_time,
        games_per_second=total_games / elapsed_time,
        steps_per_second=total_steps / elapsed_time,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark all game variants")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for benchmarks")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to run")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum steps per game")
    parser.add_argument("--warmup-batches", type=int, default=2, help="Number of warmup batches")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation, skip benchmarks")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation, only run benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode: small batch size, few batches")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    if args.quick:
        args.batch_size = 50
        args.num_batches = 2

    print("=" * 80)
    print("PGX Variant Benchmark")
    print("=" * 80)
    print(f"Git Commit: {get_git_commit_short()}")
    print(f"Device(s): {get_device_info()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num batches: {args.num_batches}")
    print("=" * 80)
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_short(),
        "device": get_device_info(),
        "jax_version": jax.__version__,
        "config": {
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "max_steps": args.max_steps,
        },
        "validation": {},
        "benchmarks": {},
    }

    # ==========================================================================
    # Validation
    # ==========================================================================
    if not args.skip_validation:
        print("=" * 80)
        print("VALIDATION")
        print("=" * 80)

        print("\n2048 Variants:")
        variants_2048 = get_2048_variants()
        for name, (EnvClass, StateClass) in variants_2048.items():
            if name == "original":
                results["validation"][f"2048_{name}"] = True
                print(f"  Validating 2048 {name}... SKIPPED (baseline)")
            else:
                valid = validate_2048_variant(name, EnvClass, StateClass)
                results["validation"][f"2048_{name}"] = valid

        print("\nBackgammon Variants:")
        variants_bg = get_backgammon_variants()
        for name, (EnvClass, StateClass) in variants_bg.items():
            if name == "original":
                results["validation"][f"backgammon_{name}"] = True
                print(f"  Validating Backgammon {name}... SKIPPED (baseline)")
            else:
                valid = validate_backgammon_variant(name, EnvClass, StateClass)
                results["validation"][f"backgammon_{name}"] = valid

        # Check for validation failures
        all_valid = all(results["validation"].values())
        if not all_valid:
            print("\nWARNING: Some validations failed!")
            if args.validate_only:
                return
        else:
            print("\nAll validations passed!")

    if args.validate_only:
        print("\nValidation only mode - skipping benchmarks")
        return

    # ==========================================================================
    # Benchmarks
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARKS")
    print("=" * 80)

    benchmark_results: List[BenchmarkResult] = []

    print("\n2048 Variants:")
    print("-" * 60)
    variants_2048 = get_2048_variants()
    for name, (EnvClass, StateClass) in variants_2048.items():
        print(f"  Benchmarking 2048 {name}...", end=" ", flush=True)
        result = benchmark_2048_variant(
            name, EnvClass, StateClass,
            args.batch_size, args.num_batches, args.max_steps, args.warmup_batches
        )
        benchmark_results.append(result)
        print(f"{result.games_per_second:,.1f} games/sec, {result.steps_per_second:,.1f} steps/sec")
        results["benchmarks"][f"2048_{name}"] = {
            "games_per_second": result.games_per_second,
            "steps_per_second": result.steps_per_second,
            "elapsed_time": result.elapsed_time,
        }

    print("\nBackgammon Variants:")
    print("-" * 60)
    variants_bg = get_backgammon_variants()
    for name, (EnvClass, StateClass) in variants_bg.items():
        print(f"  Benchmarking Backgammon {name}...", end=" ", flush=True)
        result = benchmark_backgammon_variant(
            name, EnvClass, StateClass,
            args.batch_size, args.num_batches, args.max_steps, args.warmup_batches
        )
        benchmark_results.append(result)
        print(f"{result.games_per_second:,.1f} games/sec, {result.steps_per_second:,.1f} steps/sec")
        results["benchmarks"][f"backgammon_{name}"] = {
            "games_per_second": result.games_per_second,
            "steps_per_second": result.steps_per_second,
            "elapsed_time": result.elapsed_time,
        }

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n2048 Performance Comparison:")
    print(f"{'Variant':<20} {'Games/sec':>15} {'Steps/sec':>15} {'Speedup':>10}")
    print("-" * 60)
    baseline_2048 = results["benchmarks"]["2048_original"]["games_per_second"]
    for name in variants_2048.keys():
        data = results["benchmarks"][f"2048_{name}"]
        speedup = data["games_per_second"] / baseline_2048
        print(f"{name:<20} {data['games_per_second']:>15,.1f} {data['steps_per_second']:>15,.1f} {speedup:>9.2f}x")

    print("\nBackgammon Performance Comparison:")
    print(f"{'Variant':<20} {'Games/sec':>15} {'Steps/sec':>15} {'Speedup':>10}")
    print("-" * 60)
    baseline_bg = results["benchmarks"]["backgammon_original"]["games_per_second"]
    for name in variants_bg.keys():
        data = results["benchmarks"][f"backgammon_{name}"]
        speedup = data["games_per_second"] / baseline_bg
        print(f"{name:<20} {data['games_per_second']:>15,.1f} {data['steps_per_second']:>15,.1f} {speedup:>9.2f}x")

    print("=" * 80)

    # Save results
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
