#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Naive Sudoku Solver
//  Strategy: one CUDA thread per puzzle, pure sequential
//  backtracking – direct GPU port of a CPU solver.
//  High warp divergence, no shared memory, no bitmasks.
//  This is our baseline kernel.
// ============================================================

__device__ bool naive_is_valid(const int* board, int pos, int val) {
    int row = pos / 9, col = pos % 9;
    int box_r = (row / 3) * 3, box_c = (col / 3) * 3;
    for (int i = 0; i < 9; i++) {
        if (board[row * 9 + i]            == val) return false;
        if (board[i   * 9 + col]          == val) return false;
        if (board[(box_r + i/3)*9 + (box_c + i%3)] == val) return false;
    }
    return true;
}

__device__ bool naive_solve(int* board) {
    int pos = -1;
    for (int i = 0; i < 81; i++) if (board[i] == 0) { pos = i; break; }
    if (pos == -1) return true;
    for (int val = 1; val <= 9; val++) {
        if (naive_is_valid(board, pos, val)) {
            board[pos] = val;
            if (naive_solve(board)) return true;
            board[pos] = 0;
        }
    }
    return false;
}

__global__ void naive_sudoku_kernel(const int* __restrict__ d_puzzles, int*  __restrict__ d_solutions, bool* __restrict__ d_solved, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int board[81];
    #pragma unroll
    for (int i = 0; i < 81; i++) board[i] = d_puzzles[tid * 81 + i];

    d_solved[tid] = naive_solve(board);

    #pragma unroll
    for (int i = 0; i < 81; i++) d_solutions[tid * 81 + i] = board[i];
}

inline void launch_naive(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 64) {
    naive_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
