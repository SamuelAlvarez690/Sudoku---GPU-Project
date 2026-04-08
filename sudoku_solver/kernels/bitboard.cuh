#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Bitboard Sudoku Solver  [TODO]
//  Strategy: encode row/col/box constraints as 9-bit bitmasks.
//  candidate(cell) = ~(row_mask[r] | col_mask[c] | box_mask[b]) & 0x1FF
//  Validity check = single AND vs. 27 array reads in naive.
// ============================================================

__global__ void bitboard_sudoku_kernel(const int* __restrict__ d_puzzles, int* __restrict__ d_solutions, bool* __restrict__ d_solved, int N) { 
    /* TODO */ 
}

inline void launch_bitboard(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 128) {
    bitboard_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
