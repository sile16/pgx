# Metal Testing Plan for PGX Backgammon

## Overview

This plan is for the engineer testing on Mac M1 with Metal backend. The goal is to identify which specific `lax.cond` patterns fail on Metal so we can apply minimal, targeted fixes to backgammon only.

## Prerequisites

1. Mac with Apple Silicon (M1/M2/M3)
2. Python 3.10+
3. JAX with Metal support installed:
   ```bash
   pip install jax jax-metal
   ```

## Step 1: Run the Diagnostic Script

```bash
cd /path/to/pgx
JAX_PLATFORMS=METAL python scripts/test_metal_compatibility.py
```

This script tests:
1. Basic JAX operations on Metal
2. Isolated `lax.cond` patterns to pinpoint the bug
3. PGX backgammon init/step functions
4. Internal backgammon functions

**Expected output**: A report showing which specific tests pass/fail.

## Step 2: Report Results

Please copy the full output and note:
1. Which Section 2 tests (lax.cond patterns) fail?
2. Which Section 4/5 tests (PGX environments) fail?
3. The exact error message for failures

## Step 3: Specific Function Tests (if needed)

If the diagnostic script doesn't isolate the issue, run these targeted tests:

### Test A: Core step function
```python
import os
os.environ['JAX_PLATFORMS'] = 'METAL'

import jax
import pgx

print(f"Backend: {jax.default_backend()}")

env = pgx.make("backgammon")
key = jax.random.PRNGKey(42)
state = env.init(key)
print("init() succeeded")

# This is where failure likely occurs
import jax.numpy as jnp
legal = jnp.where(state.legal_action_mask)[0]
action = legal[0]
key, subkey = jax.random.split(key)

try:
    state2 = env.step(state, action, subkey)
    print("step() succeeded")
except Exception as e:
    print(f"step() FAILED: {e}")
```

### Test B: Isolate lax.cond with State.replace
```python
import os
os.environ['JAX_PLATFORMS'] = 'METAL'

import jax
import jax.numpy as jnp
from jax import lax
import pgx.backgammon as bg

env = bg.Backgammon()
key = jax.random.PRNGKey(42)
state = env.init(key)

# Test the specific pattern from _update_by_action
@jax.jit
def test_cond_replace(state, is_no_op):
    return lax.cond(
        is_no_op,
        lambda: state,
        lambda: state.replace(terminated=jnp.array(True)),
    )

try:
    result = test_cond_replace(state, True)
    print("lax.cond with state.replace(terminated=...) succeeded")
except Exception as e:
    print(f"FAILED: {e}")
```

### Test C: Test jnp.where workaround
```python
import os
os.environ['JAX_PLATFORMS'] = 'METAL'

import jax
import jax.numpy as jnp
import pgx.backgammon as bg

env = bg.Backgammon()
key = jax.random.PRNGKey(42)
state = env.init(key)

@jax.jit
def test_where_replace(state, condition):
    new_terminated = jnp.where(condition, state.terminated, jnp.array(True))
    return state.replace(terminated=new_terminated)

try:
    result = test_where_replace(state, True)
    print("jnp.where workaround succeeded")
except Exception as e:
    print(f"FAILED: {e}")
```

## Expected Findings

Based on the bug description, we expect:
- `lax.cond` returning State with boolean fields → FAILS
- `jnp.where` on individual fields → WORKS
- Basic JAX operations → WORKS

## Locations to Fix (Backgammon Only)

Once we confirm which patterns fail, we'll apply fixes to these specific locations in `pgx/backgammon.py`:

| Line | Function | lax.cond Purpose |
|------|----------|------------------|
| 224 | `_step()` | Branch on `_is_all_off` → winning vs non-winning |
| 288 | `_no_winning_step()` | Branch on turn end → change turn vs keep state |
| 306 | `_update_by_action()` | Branch on no-op → skip update vs apply move |
| 695 | `_get_forced_single_move_mask()` | Branch on can_play_h |
| 723 | `_apply_special_backgammon_rules()` | Branch on can_play_both |
| 760 | `_legal_action_mask()` | Branch on start of turn |
| 768 | `_legal_action_mask()` | Branch on legal_action_exists |
| 807 | `_get_abs_board()` | Branch on turn (for visualization) |

Note: `pgx/core.py` also has 3 `lax.cond` calls in `Env.step()` that affect all games, but we'll only fix backgammon for now.

## After Testing

Please report back:
1. Full output of `test_metal_compatibility.py`
2. Which specific functions/patterns fail
3. Whether the `jnp.where` workaround succeeds

We'll then implement the minimal fix based on your findings.
