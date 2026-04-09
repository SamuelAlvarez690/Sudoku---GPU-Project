#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Warp-Parallel Sudoku Solver  [TODO] Jason
//  Strategy: assign 1 warp (32 threads) per puzzle.
//  Threads cooperate on constraint checking via warp shuffles.
//  Key intrinsics: __ballot_sync, __any_sync, __shfl_sync
// ============================================================

__global__ void warp_sudoku_kernel(const int* __restrict__ d_puzzles, int* __restrict__ d_solutions, bool* __restrict__ d_solved, int N) {
    // warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32
    // lane_id = threadIdx.x % 32
    /* TODO */
}

inline void launch_warp(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 256)
{
    // 32 threads per puzzle
    int total = N * 32;
    warp_sudoku_kernel<<<(total+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
