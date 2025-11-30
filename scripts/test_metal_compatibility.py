#!/usr/bin/env python3
"""
Metal Compatibility Test Script for PGX

This script tests JAX Metal backend compatibility with PGX environments.
It identifies which specific lax.cond patterns fail on Metal.

Usage:
    # On Mac with Metal:
    JAX_PLATFORMS=METAL python scripts/test_metal_compatibility.py

    # Or just run directly (will auto-detect Metal on Mac):
    python scripts/test_metal_compatibility.py

The script will:
1. Test basic JAX Metal functionality
2. Test isolated lax.cond patterns to identify the bug
3. Test each PGX environment's init/step functions
4. Report which specific operations fail
"""

import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Any

# Try to use Metal on Mac
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORMS", "METAL")

import jax
import jax.numpy as jnp
from jax import lax


def test_result(name: str, passed: bool, error: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if error and not passed:
        # Print first line of error
        first_line = error.split('\n')[0][:80]
        print(f"         Error: {first_line}")


def run_test(name: str, fn: Callable[[], Any]) -> bool:
    """Run a test function and report results."""
    try:
        fn()
        test_result(name, True)
        return True
    except Exception as e:
        test_result(name, False, str(e))
        return False


# =============================================================================
# Section 1: Basic JAX Metal Tests
# =============================================================================

def test_basic_jax():
    """Test basic JAX operations work on Metal."""
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    assert z.shape == (3,)
    assert float(z[0]) == 5.0


def test_jit_basic():
    """Test JIT compilation works."""
    @jax.jit
    def add(x, y):
        return x + y

    result = add(jnp.array(1.0), jnp.array(2.0))
    assert float(result) == 3.0


def test_vmap_basic():
    """Test vmap works."""
    def square(x):
        return x * x

    xs = jnp.array([1.0, 2.0, 3.0])
    result = jax.vmap(square)(xs)
    assert result.shape == (3,)


# =============================================================================
# Section 2: lax.cond Pattern Tests - Isolate the Bug
# =============================================================================

def test_cond_scalar_return():
    """Test lax.cond returning scalar."""
    def fn(cond):
        return lax.cond(cond, lambda: 1.0, lambda: 2.0)

    result = jax.jit(fn)(True)
    assert float(result) == 1.0


def test_cond_array_return():
    """Test lax.cond returning array."""
    def fn(cond):
        return lax.cond(
            cond,
            lambda: jnp.array([1.0, 2.0]),
            lambda: jnp.array([3.0, 4.0])
        )

    result = jax.jit(fn)(True)
    assert result.shape == (2,)


def test_cond_bool_return():
    """Test lax.cond returning boolean - LIKELY TO FAIL ON METAL."""
    def fn(cond):
        return lax.cond(cond, lambda: True, lambda: False)

    result = jax.jit(fn)(True)
    assert bool(result) == True


def test_cond_bool_array_return():
    """Test lax.cond returning boolean array - LIKELY TO FAIL ON METAL."""
    def fn(cond):
        return lax.cond(
            cond,
            lambda: jnp.array([True, False]),
            lambda: jnp.array([False, True])
        )

    result = jax.jit(fn)(True)
    assert result.shape == (2,)


def test_cond_tuple_with_bool():
    """Test lax.cond returning tuple containing boolean - LIKELY TO FAIL ON METAL."""
    def fn(cond):
        return lax.cond(
            cond,
            lambda: (jnp.array(1.0), jnp.array(True)),
            lambda: (jnp.array(2.0), jnp.array(False))
        )

    result = jax.jit(fn)(True)
    assert len(result) == 2


# Test with frozen dataclass (similar to PGX State)
@dataclass(frozen=True)
class SimpleState:
    value: Any
    flag: Any  # boolean field


def test_cond_dataclass_replace_bool():
    """Test lax.cond with dataclass.replace modifying bool - LIKELY TO FAIL ON METAL."""
    import dataclasses

    def fn(state, cond):
        return lax.cond(
            cond,
            lambda: dataclasses.replace(state, flag=jnp.array(True)),
            lambda: dataclasses.replace(state, flag=jnp.array(False))
        )

    initial = SimpleState(value=jnp.array(1.0), flag=jnp.array(False))
    result = jax.jit(fn)(initial, True)
    assert bool(result.flag) == True


def test_cond_dataclass_replace_non_bool():
    """Test lax.cond with dataclass.replace NOT modifying bool - may work on Metal."""
    import dataclasses

    def fn(state, cond):
        return lax.cond(
            cond,
            lambda: dataclasses.replace(state, value=jnp.array(10.0)),
            lambda: dataclasses.replace(state, value=jnp.array(20.0))
        )

    initial = SimpleState(value=jnp.array(1.0), flag=jnp.array(False))
    result = jax.jit(fn)(initial, True)
    assert float(result.value) == 10.0


# =============================================================================
# Section 3: jnp.where Alternative Tests
# =============================================================================

def test_where_bool():
    """Test jnp.where with boolean - should work on Metal."""
    def fn(cond):
        return jnp.where(cond, jnp.array(True), jnp.array(False))

    result = jax.jit(fn)(True)
    assert bool(result) == True


def test_where_dataclass_fields():
    """Test jnp.where for individual dataclass fields - should work on Metal."""
    import dataclasses

    def fn(state, cond):
        new_value = jnp.where(cond, jnp.array(10.0), jnp.array(20.0))
        new_flag = jnp.where(cond, jnp.array(True), jnp.array(False))
        return dataclasses.replace(state, value=new_value, flag=new_flag)

    initial = SimpleState(value=jnp.array(1.0), flag=jnp.array(False))
    result = jax.jit(fn)(initial, True)
    assert float(result.value) == 10.0
    assert bool(result.flag) == True


# =============================================================================
# Section 4: PGX Environment Tests
# =============================================================================

def test_pgx_backgammon_init():
    """Test Backgammon environment init."""
    import pgx.backgammon as bg
    env = bg.Backgammon()
    key = jax.random.PRNGKey(42)
    state = env.init(key)
    assert state is not None
    assert hasattr(state, 'terminated')


def test_pgx_backgammon_step():
    """Test Backgammon environment step - LIKELY TO FAIL ON METAL."""
    import pgx.backgammon as bg
    env = bg.Backgammon()
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    # Find a legal action
    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        key, subkey = jax.random.split(key)
        new_state = env.step(state, action, subkey)
        assert new_state is not None


def test_pgx_backgammon_step_jit():
    """Test JIT-compiled Backgammon step - LIKELY TO FAIL ON METAL."""
    import pgx.backgammon as bg
    env = bg.Backgammon()

    @jax.jit
    def do_step(state, action, key):
        return env.step(state, action, key)

    key = jax.random.PRNGKey(42)
    state = env.init(key)

    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        key, subkey = jax.random.split(key)
        new_state = do_step(state, action, subkey)
        assert new_state is not None


def test_pgx_core_step():
    """Test core.Env.step directly - LIKELY TO FAIL ON METAL."""
    import pgx
    env = pgx.make("tic_tac_toe")  # Simpler environment
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    # Find a legal action
    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        new_state = env.step(state, action)
        assert new_state is not None


def test_pgx_connect_four():
    """Test Connect Four environment."""
    import pgx
    env = pgx.make("connect_four")
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        new_state = env.step(state, action)
        assert new_state is not None


def test_pgx_othello():
    """Test Othello environment."""
    import pgx
    env = pgx.make("othello")
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        new_state = env.step(state, action)
        assert new_state is not None


# =============================================================================
# Section 5: Specific Backgammon lax.cond Tests
# =============================================================================

def test_backgammon_step_internal():
    """Test backgammon._step directly."""
    import pgx.backgammon as bg

    env = bg.Backgammon()
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        key, subkey = jax.random.split(key)
        # Call internal _step
        new_state = bg._step(state, action, subkey)
        assert new_state is not None


def test_backgammon_no_winning_step():
    """Test backgammon._no_winning_step directly."""
    import pgx.backgammon as bg

    env = bg.Backgammon()
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        key, subkey = jax.random.split(key)
        new_state = bg._no_winning_step(state, action, subkey)
        assert new_state is not None


def test_backgammon_update_by_action():
    """Test backgammon._update_by_action directly."""
    import pgx.backgammon as bg

    env = bg.Backgammon()
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        new_state = bg._update_by_action(state, action)
        assert new_state is not None


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print("=" * 70)
    print("PGX JAX Metal Compatibility Test")
    print("=" * 70)
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    if jax.default_backend() != "METAL":
        print("WARNING: Not running on Metal backend!")
        print("On Mac, set JAX_PLATFORMS=METAL to test Metal compatibility.")
        print()

    results = {"pass": 0, "fail": 0}

    def run_section(name: str, tests: list):
        print(f"\n{name}")
        print("-" * 60)
        for test_fn in tests:
            passed = run_test(test_fn.__name__, test_fn)
            results["pass" if passed else "fail"] += 1

    # Section 1: Basic JAX
    run_section("Section 1: Basic JAX Operations", [
        test_basic_jax,
        test_jit_basic,
        test_vmap_basic,
    ])

    # Section 2: lax.cond patterns
    run_section("Section 2: lax.cond Pattern Tests (Bug Isolation)", [
        test_cond_scalar_return,
        test_cond_array_return,
        test_cond_bool_return,
        test_cond_bool_array_return,
        test_cond_tuple_with_bool,
        test_cond_dataclass_replace_bool,
        test_cond_dataclass_replace_non_bool,
    ])

    # Section 3: jnp.where alternatives
    run_section("Section 3: jnp.where Alternative Tests", [
        test_where_bool,
        test_where_dataclass_fields,
    ])

    # Section 4: PGX environments
    run_section("Section 4: PGX Environment Tests", [
        test_pgx_backgammon_init,
        test_pgx_backgammon_step,
        test_pgx_backgammon_step_jit,
        test_pgx_core_step,
        test_pgx_connect_four,
        test_pgx_othello,
    ])

    # Section 5: Backgammon internals
    run_section("Section 5: Backgammon Internal Function Tests", [
        test_backgammon_step_internal,
        test_backgammon_no_winning_step,
        test_backgammon_update_by_action,
    ])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = results["pass"] + results["fail"]
    print(f"Passed: {results['pass']}/{total}")
    print(f"Failed: {results['fail']}/{total}")

    if results["fail"] > 0:
        print("\nFailed tests indicate Metal compatibility issues.")
        print("Look for patterns in failures to identify the root cause.")
        print("\nCommon pattern: lax.cond with boolean returns fails on Metal.")
        print("Fix: Replace lax.cond with jnp.where for field-by-field selection.")
    else:
        print("\nAll tests passed! PGX appears compatible with Metal backend.")

    return results["fail"]


if __name__ == "__main__":
    sys.exit(main())
