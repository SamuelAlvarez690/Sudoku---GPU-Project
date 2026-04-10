#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Bitboard Sudoku Solver  [TODO] Jakob
//  Strategy: encode row/col/box constraints as 9-bit bitmasks.
//  candidate(cell) = ~(row_mask[r] | col_mask[c] | box_mask[b]) & 0x1FF
//  Validity check = single AND vs. 27 array reads in naive.
// ============================================================

// Compute which 3x3 box a cell belongs to
__device__ __forceinline__ int box_id(int r, int c) {
    return (r / 3) * 3 + (c / 3);
}

// Count how many bits are set (how many candidates exist)
__device__ __forceinline__ int popcount9(unsigned short x) {
    return __popc((unsigned int)x);
}

// Recursive solver using bitmasks
__device__ bool bitboard_solve(int* board, unsigned short row_mask[9], unsigned short col_mask[9], unsigned short box_mask[9]) {
    int best_pos = -1;              // position of best empty cell
    unsigned short best_cand = 0;   // candidate bitmask
    int best_count = 10;            // minimum candidates found

    // Find the best empty cell using MRV (Minimum Remaining Values)
    // This reduces branching and speeds up solving
    for (int pos = 0; pos < 81; pos++) {
        if (board[pos] != 0) continue; // skips filled cells

        // Convert 1D array index into Sudoku coordinates
        int r = pos / 9;
        int c = pos % 9;
        int b = box_id(r, c);

        // Combine number constraints from row, column, and box
        unsigned short used = row_mask[r] | col_mask[c] | box_mask[b];

        // Invert all bits to give you which numbers are still allowed (only keep 9 bits -> 0x1FF)
        unsigned short cand = (~used) & 0x1FF;

        // Count how many valid options exist
        int cnt = popcount9(cand);

        // If no candidates then dead end then backtrack
        if (cnt == 0) return false;

        // Choose cell with fewest candidates
        if (cnt < best_count) {
            best_count = cnt;
            best_pos = pos;
            best_cand = cand;

            // If only 1 option, stop searching
            if (cnt == 1) break;
        }
    }

    // If no empty cells remain then puzzle is solved
    if (best_pos == -1) return true;

    int r = best_pos / 9;
    int c = best_pos % 9;
    int b = box_id(r, c);

    // Loop through all possible values
    while (best_cand) {
        // Extract lowest set bit (fast candidate iteration) 
        unsigned short bit = best_cand & (unsigned short)(-((short)best_cand));

        // Convert bit to digit (1-9)
        int digit = __ffs((int)bit);

        // Place digit
        board[best_pos] = digit;

        // Update constraints
        row_mask[r] |= bit;
        col_mask[c] |= bit;
        box_mask[b] |= bit;

        // Recursive solver
        if (bitboard_solve(board, row_mask, col_mask, box_mask))
            return true;
        
        // Undo move (backtrack)
        board[best_pos] = 0;
        row_mask[r] &= ~bit;
        col_mask[c] &= ~bit;
        box_mask[b] &= ~bit;

        // Remove this candidate and try next
        best_cand &= (best_cand - 1);
    }

    return false; // No solution found at this branch
}

__global__ void bitboard_sudoku_kernel(const int* __restrict__ d_puzzles, int* __restrict__ d_solutions, bool* __restrict__ d_solved, int N) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // One thread = one puzzle
    if (tid >= N) return;

    int board[81];

    // Bitmasks:
    // Each bit represents digits 1-9 (bit 0 = 1, bit 8 = 9)
    unsigned short row_mask[9] = {0};
    unsigned short col_mask[9] = {0};
    unsigned short box_mask[9] = {0};

    bool valid = true;

    // Copy puzzle and initialize constraint masks
    #pragma unroll
    for (int i = 0; i < 81; i++) {
        // Read from global memory
        int v = d_puzzles[tid * 81 + i];
        board[i] = v;

        // Only process filled cells
        if (v != 0) {
            int r = i / 9;
            int c = i % 9;
            int b = box_id(r, c);

            // Convert number to bit
            unsigned short bit = (unsigned short)(1u << (v - 1));

            // If already used then invalid puzzle
            if ((row_mask[r] & bit) || (col_mask[c] & bit) || (box_mask[b] & bit)) {
                valid = false;
            } else {
                // Mark digit as used
                row_mask[r] |= bit;
                col_mask[c] |= bit;
                box_mask[b] |= bit;
            }
        }
    }

    // Solve puzzle if valid
    bool solved = false;
    
    // Solve if initial board is valid
    if (valid) {
        solved = bitboard_solve(board, row_mask, col_mask, box_mask);
    }

    d_solved[tid] = solved;

    // Copy result back to global memory
    #pragma unroll
    for (int i = 0; i < 81; i++) {
        d_solutions[tid * 81 + i] = board[i];
    }
}

inline void launch_bitboard(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 128) {
    bitboard_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
