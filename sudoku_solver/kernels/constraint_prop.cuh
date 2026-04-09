#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Constraint Propagation Solver  [TODO] Sam
//  Strategy: run naked-singles pass before backtracking.
//  Cells with only one candidate are filled immediately.
//  Easy/medium puzzles may solve with zero backtracking.
// ============================================================

__global__ void cp_sudoku_kernel(const int* __restrict__ d_puzzles, int* __restrict__ d_solutions, bool* __restrict__ d_solved, int N) { 
    /* TODO */ 
}

inline void launch_cp(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 64)
{
    cp_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
