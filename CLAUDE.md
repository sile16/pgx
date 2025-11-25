# CLAUDE.md

## Project Overview

PGX is a collection of GPU-accelerated board game environments built on JAX. This document tracks our current development focus.

## Current Goal: Backgammon Performance Optimization

We are refactoring the PGX Backgammon implementation and tracking performance changes across git commits. The benchmark measures games per second to detect improvements or regressions.

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
- Steps per second (total game steps processed)
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
# Full benchmark with JSON output (recommended for tracking)
python benchmarks/benchmark_backgammon.py --batch-sizes 1,10,100,1000,10000 --output-json benchmarks/benchmark_results.json

# Quick test (for development)
python benchmarks/benchmark_backgammon.py --batch-sizes 100,1000 --num-batches 3

# With custom parameters
python benchmarks/benchmark_backgammon.py --batch-sizes 100,1000 --num-batches 10 --max-steps 5000
```

### Benchmark Options
- `--batch-sizes`: Comma-separated list of batch sizes to test (default: 1,10,100,1000,10000)
- `--num-batches`: Number of batches to run per batch size (default: 10)
- `--max-steps`: Maximum steps per game (default: 5000)
- `--warmup-batches`: Number of warmup batches for JIT compilation (default: 2)
- `--short-game`: Use short game variant (pieces start halfway through)
- `--output-json`: Path to JSON file to save results (appends to existing file)

## Project Structure
- `pgx/` - Core game implementations
- `pgx/backgammon.py` - Backgammon game implementation
- `tests/` - Test suite
- `benchmarks/` - Performance benchmarks
