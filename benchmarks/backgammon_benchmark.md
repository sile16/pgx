# Backgammon Performance Optimization

This document tracks performance optimizations made to the PGX Backgammon implementation.

## Hardware

- GPU: NVIDIA GeForce RTX 4090 (24GB)
- CPU: Intel Core i7-10700K @ 3.80GHz (8 cores, 16 threads)
- JAX version: 0.8.1

## Performance History

| Commit | Description | Games/sec | Improvement |
|--------|-------------|-----------|-------------|
| c14fcf8 | Baseline | 14.4 | - |
| 1181cc0 | Lookup table for action decomposition | 17.5 | +22% |
| 1181cc0 | + JAX while_loop (eliminate Python loop) | 96.6 | +5.5x |
| 0681f75 | + Focused candidate search optimization | 4,858 | +50x |
| aeb544e | + Direct scatter in `_legal_action_mask_for_valid_single_dice` | 4,171* | +12.5%** |
| Current | + Fused mask computation, hoisted constants | 5,797 | +19% |

*Note: Benchmarks use short game mode for faster iteration.

**Improvement measured against pre-optimization baseline of 3,706 games/sec in current benchmark setup.

## Optimizations Implemented

### 1. Lookup Table for Action Decomposition (Refactor #3)
**Commit:** 1181cc0

Replaced runtime calculation of `src`, `die`, and `tgt` from action indices with pre-computed lookup tables. This eliminates branching and math operations during gameplay.

```python
# Before: Runtime calculation with branching
def _calc_src(action): ...
def _calc_tgt(action): ...

# After: O(1) array lookup
_ACTION_SRC_LOOKUP, _ACTION_DIE_LOOKUP, _ACTION_TGT_LOOKUP = _build_lookup_tables()
```

### 2. JAX while_loop (Refactor #2)
**Commit:** 1181cc0

Replaced Python for-loops with `jax.lax.while_loop` to eliminate Python interpreter overhead and enable full XLA compilation.

### 3. Focused Candidate Search (Refactor #4)
**Commit:** 0681f75

In `_get_valid_sequence_mask`, reduced computation from 156 actions to 26 by only processing actions that match the current die value. Previously wasted 83% of compute on irrelevant actions.

```python
# Before: Process all 156 actions
all_actions = jnp.arange(26 * 6)
next_boards = jax.vmap(_move)(board, all_actions)  # Shape: (156, 28)

# After: Process only 26 relevant actions
candidate_actions = src_indices * 6 + (die_first - 1)  # Shape: (26,)
next_boards = jax.vmap(_move)(board, candidate_actions)  # Shape: (26, 28)
```

### 4. Direct Scatter in Legal Action Mask (Refactor #5)
**Commit:** Current

In `_legal_action_mask_for_valid_single_dice`, changed from creating a (26, 156) intermediate array to directly scattering 26 results into a 156-element mask.

```python
# Before: Create 26 full 156-element masks, then reduce
def _is_legal(idx):
    mask = jnp.zeros(156)
    mask = mask.at[idx * 6 + die].set(_is_action_legal(board, idx * 6 + die))
    return mask
jax.vmap(_is_legal)(src_indices).any(axis=0)  # (26, 156) -> (156,)

# After: Direct scatter
actions = src_indices * 6 + die  # Shape: (26,)
is_legal = jax.vmap(_is_action_legal)(board, actions)  # Shape: (26,)
mask = jnp.zeros(156).at[actions].set(is_legal)
```

### 5. Histogram-style Dice Counting (Refactor #6)
**Commit:** Current

Simplified `_to_playable_dice_count` to use histogram-style counting instead of vmap + tile.

### 6. Simplified Playable Dice Update (Refactor #7)
**Commit:** Current

Simplified `_update_playable_dice` using `jnp.where` and `jnp.argmax` instead of vmap + tile.

### 7. Simplified Set Playable Dice (Refactor #8)
**Commit:** aeb544e

Simplified `_set_playable_dice` using `jnp.where` instead of arithmetic with array literals.

### 8. Fused Mask Computation (Refactor #9)
**Commit:** Current

Fused `_can_play_two_dice` and `_get_forced_full_move_mask` into a single `_compute_two_dice_masks` function. Previously, `_apply_special_backgammon_rules` would call both functions which each computed the same two sequence masks independently (4 calls to `_get_valid_sequence_mask`). Now computes both masks once (2 calls).

```python
# Before: Redundant computation
can_play_both = _can_play_two_dice(board, d1, d2)  # 2 calls to _get_valid_sequence_mask
if can_play_both:
    mask = _get_forced_full_move_mask(board, d1, d2)  # 2 MORE calls (same masks!)

# After: Compute once, reuse
mask_d1_d2, mask_d2_d1, can_play_both = _compute_two_dice_masks(board, d1, d2)  # 2 calls total
forced_full_move_mask = mask_d1_d2 | mask_d2_d1  # Reuse computed masks
```

### 9. Hoisted Constant Array Allocations (Refactor #10)
**Commit:** Current

Hoisted frequently-used constant arrays to module level:
- `_SRC_INDICES = jnp.arange(26, dtype=jnp.int32)` - used in 3 functions
- `_NO_OP_MASK` - the no-op legal action mask

This avoids repeated array allocation in JIT-compiled code.

## Future Optimization Opportunities

### High Priority

~~1. **Hoist Constant Array Allocations** - DONE (Refactor #10)~~

~~2. **Fuse Operations in Special Rules** - DONE (Refactor #9)~~

~~3. **Early Exit for Doubles** - Already implemented via `jax.lax.cond`~~

### Medium Priority

4. **Vectorize `_is_action_legal` Internals**
   - Currently called via vmap, but internal logic has branching
   - Could restructure to be more vectorization-friendly

5. **Batch Board State Updates**
   - When checking future move legality, we create many board states
   - Could potentially batch these operations more efficiently

### Low Priority

6. **Profile Memory Allocation Patterns**
   - Use JAX profiler to identify allocation hotspots
   - May reveal opportunities for buffer reuse

7. **Explore Alternative Data Structures**
   - Current board representation uses 28-element array
   - Could explore bitboard representations for certain operations

## CPU vs GPU Performance Comparison

For ML integration where backgammon may interact with other game engines, understanding CPU vs GPU tradeoffs is important.

### GPU Performance (RTX 4090)

| Batch Size | Games/sec | Steps/sec | Moves/sec |
|------------|-----------|-----------|-----------|
| 1,000 | 4,164 | 647,295 | 458,008 |
| 2,000 | 5,563 | 864,661 | 611,872 |
| **4,000** | **5,797** | **900,543** | **637,405** |

**Optimal batch size: 4,000** (best throughput)

### CPU Performance (i7-10700K)

| Batch Size | Games/sec | Steps/sec | Moves/sec |
|------------|-----------|-----------|-----------|
| 1 | 47 | 7,793 | 5,472 |
| 10 | 83 | 12,753 | 9,028 |
| 25 | 95 | 14,969 | 10,579 |
| 50 | 102 | 16,059 | 11,354 |
| **100** | **121** | **18,887** | **13,356** |
| 200 | 99 | 15,736 | 11,141 |

**Optimal batch size: 100** (best throughput)

### Summary

| Device | Optimal Batch | Games/sec | GPU Speedup |
|--------|---------------|-----------|-------------|
| GPU (RTX 4090) | 4,000 | 5,797 | 48x |
| CPU (i7-10700K) | 100 | 121 | 1x (baseline) |

### Recommendations for ML Integration

| Use Case | Recommended Device | Batch Size |
|----------|-------------------|------------|
| Interactive/low latency | CPU | 1-10 |
| Small batch RL training | CPU | 50-100 |
| Large batch training | GPU | 2,000-4,000 |
| Integration with sequential engines | CPU | 1-10 |

**Key insight:** The 48x GPU advantage only materializes at high batch sizes. If your ML pipeline is bottlenecked by other engines processing one game at a time, CPU with small batches may be preferable to avoid GPU transfer overhead.

## Benchmark Commands

```bash
# Quick GPU benchmark (~1 min, recommended for development)
python benchmarks/benchmark_backgammon.py --quick

# Full GPU benchmark with multiple batch sizes
python benchmarks/benchmark_backgammon.py --batch-sizes 1000,2000,4000 --num-batches 3

# CPU benchmark (set JAX_PLATFORMS before running)
JAX_PLATFORMS=cpu python benchmarks/benchmark_backgammon.py --batch-sizes 1,10,50,100,200 --num-batches 5 --short-game

# Save results to JSON
python benchmarks/benchmark_backgammon.py --quick --output-json benchmarks/benchmark_results.json
```
