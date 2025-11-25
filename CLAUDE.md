# CLAUDE.md

## Project Overview

PGX is a collection of GPU-accelerated board game environments built on JAX. This document tracks our current development focus.

## Current Goal: Backgammon Performance Optimization

We are refactoring the PGX Backgammon implementation and tracking performance changes across git commits. The benchmark measures games per second to detect improvements or regressions.

### Performance Results

| Commit | Description | Games/sec | Steps/sec | Moves/sec |
|--------|-------------|-----------|-----------|-----------|
| c14fcf8 | Baseline | 14.4 | 2,233 | - |
| 1181cc0 | Lookup table for action decomposition | 17.5 | 2,728 | - |
| 1181cc0 | + JAX while_loop (eliminate Python loop) | 96.6 | 15,014 | 10,623 |
| 0681f75 | + Focused candidate search optimization | 4,858 | 755,149 | 534,377 |
| Current | + Fused mask computation, hoisted constants | 5,797 | 900,543 | 637,405 |

**Total improvement: 402x faster** (14.4 → 5,797 games/sec)

### Key Optimizations

1. **Lookup table for action decomposition** (+22% speedup)
   - Move src/tgt calculation from runtime to precomputed table

2. **JAX while_loop** (+5.5x speedup)
   - Replace Python for-loop with `jax.lax.while_loop`
   - Eliminates Python interpreter overhead

3. **Focused candidate search** (+50x speedup)
   - `_get_valid_sequence_mask`: Only process 26 relevant actions instead of 156
   - Reduces wasted computation by 83%

4. **Fused mask computation** (+19% speedup)
   - Fuse `_can_play_two_dice` and `_get_forced_full_move_mask` to avoid redundant computation
   - Hoist constant array allocations (`_SRC_INDICES`, `_NO_OP_MASK`) to module level

### Why We're Benchmarking
- Track performance impact of code refactors
- Compare results across different git commits
- Detect regressions early
- Find optimal batch sizes for GPU utilization

### Benchmark Location
- `benchmarks/benchmark_backgammon.py` - The benchmark script
- `benchmarks/benchmark_results.json` - Historical results log (per commit)

### What the Benchmark Measures
- Games per second at various batch sizes
- Steps per second (total game steps including dice rolls)
- Moves per second (player actions only, excludes stochastic steps)
- Game statistics: avg/min/max steps, point distribution (1pt/2pt/3pt games)
- Optimal batch size for throughput on the current hardware

## Development Environment

### Virtual Environment
Always activate the virtual environment before running Python:
```bash
source venv-pgx/bin/activate
```

### CUDA/GPU Support
JAX with CUDA 12 support is installed in the venv. To install/update:
```bash
pip install --upgrade "jax[cuda12]"
```

Verify GPU is available:
```bash
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

### Running the Benchmark
```bash
# Quick benchmark (~1 min, recommended for development)
python benchmarks/benchmark_backgammon.py --quick --output-json benchmarks/benchmark_results.json

# Full benchmark with multiple batch sizes
python benchmarks/benchmark_backgammon.py --batch-sizes 1000,2000,4000 --num-batches 3 --short-game --output-json benchmarks/benchmark_results.json

# Standard benchmark
python benchmarks/benchmark_backgammon.py --batch-sizes 1,10,100,1000,10000 --output-json benchmarks/benchmark_results.json
```

### Benchmark Options
- `--batch-sizes`: Comma-separated list of batch sizes to test (default: 1,10,100,1000,10000)
- `--num-batches`: Number of batches to run per batch size (default: 10)
- `--max-steps`: Maximum steps per game (default: 5000)
- `--warmup-batches`: Number of warmup batches for JIT compilation (default: 2)
- `--short-game`: Use short game variant (pieces start halfway through)
- `--quick`: Quick mode - batch size 1000, 3 batches, short game
- `--output-json`: Path to JSON file to save results (appends to existing file)

### Running Tests
```bash
pip install pytest
python -m pytest tests/test_backgammon.py -v
```

## Project Structure
- `pgx/` - Core game implementations
- `pgx/backgammon.py` - Backgammon game implementation
- `tests/` - Test suite
- `benchmarks/` - Performance benchmarks

## Hardware
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- Optimal batch size: 4000 (best throughput)
- Batch size 5000+ causes OOM
