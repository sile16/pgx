# Backgammon Performance Optimization

This document tracks performance optimizations made to the PGX Backgammon implementation.

## Hardware

- GPU: NVIDIA GeForce RTX 4090 (24GB)
- JAX version: 0.8.1

## Performance History

| Commit | Description | Games/sec | Improvement |
|--------|-------------|-----------|-------------|
| c14fcf8 | Baseline | 14.4 | - |
| 1181cc0 | Lookup table for action decomposition | 17.5 | +22% |
| 1181cc0 | + JAX while_loop (eliminate Python loop) | 96.6 | +5.5x |
| 0681f75 | + Focused candidate search optimization | 4,858 | +50x |
| Current | + Direct scatter in `_legal_action_mask_for_valid_single_dice` | 4,171* | +12.5%** |

*Note: Current benchmark uses short game mode for faster iteration; baseline 4,858 was measured differently.

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
**Commit:** Current

Simplified `_set_playable_dice` using `jnp.where` instead of arithmetic with array literals.

## Future Optimization Opportunities

### High Priority

1. **Hoist Constant Array Allocations**
   - `jnp.arange(26, dtype=jnp.int32)` is created repeatedly in multiple functions
   - Could be a module-level constant `_SRC_INDICES`
   - Potential impact: Low-medium (depends on JIT caching behavior)

2. **Fuse Operations in Special Rules**
   - `_can_play_two_dice` and `_get_forced_full_move_mask` compute overlapping work
   - Both call `_get_valid_sequence_mask` with the same arguments
   - Could cache/reuse the sequence masks
   - Potential impact: Medium (only affects non-doubles first move)

3. **Early Exit for Doubles**
   - The special backgammon rules (must play both dice, must play higher) only apply to non-doubles
   - For doubles, we could skip computing the special rule masks entirely
   - Currently handled by `jax.lax.cond` but the simple mask is still computed

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

## Benchmark Commands

```bash
# Quick benchmark (~1 min, recommended for development)
python benchmarks/benchmark_backgammon.py --quick

# Full benchmark with multiple batch sizes
python benchmarks/benchmark_backgammon.py --batch-sizes 1000,2000,4000 --num-batches 3

# Save results to JSON
python benchmarks/benchmark_backgammon.py --quick --output-json benchmarks/benchmark_results.json
```
