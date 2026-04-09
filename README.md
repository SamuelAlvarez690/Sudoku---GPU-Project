# COSC 4397 – GPU Sudoku Solver

## Team
Jason Pedder
Samuel Alvarez
Jakob Lamb

## Overview
GPU-accelerated Sudoku solver benchmarking suite.  
Each kernel solves a **batch** of puzzles in parallel (one thread / warp / block per puzzle depending on the kernel variant).

## Project Structure

```
sudoku_solver/
├── Makefile
├── sudoku_bench.cu          ← main benchmark driver
├── kernels/
│   ├── naive.cuh            ← Baseline: 1 thread per puzzle, sequential backtracking
│   ├── bitboard.cuh         ← TODO: bitmask constraint encoding implemented by Jakob
│   ├── constraint_prop.cuh  ← TODO: naked singles + backtracking implemented by Sam
│   └── warp_parallel.cuh    ← TODO: 1 warp (32 threads) per puzzle implemented by Sam
└── utils/
    ├── timer.h              ← CUDA event timer + error check macros
    └── puzzle_io.h          ← puzzle bank, loader, validator, batch generator
```

## Hardware Target
NVIDIA RTX 3080 (sm_86).  
Edit `ARCH` in `Makefile` for other GPUs.

## Build & Run

```bash
# Build
make

# Build and run all benchmarks
make run

# Clean
make clean

# Print GPU info
make info
```

## Expected Output Format

```
Device: NVIDIA GeForce RTX 3080

===================================================
Sudoku solver  --  batch size: 64 puzzles
---------------------------------------------------
Kernel                  Time (ms)    Puzzles/s   Solved
---------------------------------------------------
Naive                      12.34       5187.1   64/64
Bitboard                    3.21      19938.0   64/64
Constraint Prop             1.05      60952.4   64/64
Warp Parallel               0.88      72727.3   64/64
===================================================
```

## Kernels

| Kernel | Strategy | Status |
|--------|----------|--------|
| **Naive** | 1 thread/puzzle, sequential backtracking 
| **Bitboard** | Bitmask rows/cols/boxes, faster validity
| **Constraint Prop** | Naked singles pass before backtrack
| **Warp Parallel** | 32 threads cooperate on 1 puzzle

## Correctness Validation
- Every solution is checked with `validate_board()` (rows + cols + boxes must each contain 1–9 exactly once).
- CPU reference solver (`cpu_solve`) verifies GPU outputs match.

## Performance Metric
**Puzzles/second** = `batch_size / (avg_kernel_time_seconds)`

Batch sizes tested: **64, 256, 1024**

## Course Concepts Demonstrated
- Thread/block decomposition
- Register pressure vs. shared memory tradeoffs
- Warp divergence (branching in backtracker)
- Occupancy analysis
- Memory coalescing (batch layout)
- Constraint propagation as a pruning strategy

## Reproducibility
```bash
git clone <repo>
cd sudoku_solver
make run
```
No external dependencies beyond CUDA toolkit (≥ 11.0).
