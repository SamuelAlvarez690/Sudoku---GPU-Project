#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Constraint Propagation Solver  [TODO] Sam
//  Strategy: run naked-singles pass before backtracking.
//  Cells with only one candidate are filled immediately.
//  Easy/medium puzzles may solve with zero backtracking.
// ============================================================

__device__ inline int get_box_idx(int row, int col) {
    return (row / 3) * 3 + (col / 3);
}

__device__ bool cp_solve(int* board, int* rows, int* cols, int* boxes, int* iter) {
    if (++(*iter) > 1000000) return false; // Safety guard

    int best_pos = -1;
    int min_candidates = 10;
    int best_mask = 0;

    // Constraint Propagation & MRV Heuristic
    for (int i = 0; i < 81; i++) {
        if (board[i] == 0) {
            int r = i / 9, c = i % 9, b = get_box_idx(r, c);
            // 0x3FE is bits 1-9 set (binary 1111111110)
            int forbidden = rows[r] | cols[c] | boxes[b];
            int candidates = (~forbidden) & 0x3FE;
            
            int count = __popc(candidates); // GPU intrinsic for bit counting
            if (count == 0) return false; // Dead end

            if (count < min_candidates) {
                min_candidates = count;
                best_pos = i;
                best_mask = candidates;
            }
            if (count == 1) break; // Found a Naked Single, prioritize it
        }
    }

    if (best_pos == -1) return true; // Solved!

    int r = best_pos / 9, c = best_pos % 9, b = get_box_idx(r, c);

    for (int val = 1; val <= 9; val++) {
        if (best_mask & (1 << val)) {
            // Apply constraints
            board[best_pos] = val;
            rows[r] |= (1 << val);
            cols[c] |= (1 << val);
            boxes[b] |= (1 << val);

            if (cp_solve(board, rows, cols, boxes, iter)) return true;

            // Backtrack (Undo constraints)
            board[best_pos] = 0;
            rows[r] &= ~(1 << val);
            cols[c] &= ~(1 << val);
            boxes[b] &= ~(1 << val);
        }
    }
    return false;
}

__global__ void cp_sudoku_kernel(const int* __restrict__ d_puzzles, int* __restrict__ d_solutions, bool* __restrict__ d_solved, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int board[81];
    int rows[9] = {0}, cols[9] = {0}, boxes[9] = {0};

    // Initialize board and constraint masks
    #pragma unroll
    for (int i = 0; i < 81; i++) {
        int val = d_puzzles[tid * 81 + i];
        board[i] = val;
        if (val > 0) {
            int r = i / 9, c = i % 9;
            rows[r] |= (1 << val);
            cols[c] |= (1 << val);
            boxes[get_box_idx(r, c)] |= (1 << val);
        }
    }

    // Constraint Propagation Loop: Fill Naked Singles until no more can be found
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < 81; i++) {
            if (board[i] == 0) {
                int r = i / 9, c = i % 9, b = get_box_idx(r, c);
                int candidates = (~(rows[r] | cols[c] | boxes[b])) & 0x3FE;
                if (__popc(candidates) == 1) {
                    int val = __ffs(candidates) - 1; // Find First Set bit
                    board[i] = val;
                    rows[r] |= (1 << val);
                    cols[c] |= (1 << val);
                    boxes[b] |= (1 << val);
                    changed = true;
                }
            }
        }
    }

    int iter = 0;
    d_solved[tid] = cp_solve(board, rows, cols, boxes, &iter);

    #pragma unroll
    for (int i = 0; i < 81; i++) d_solutions[tid * 81 + i] = board[i];
}
inline void launch_cp(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 64)
{
    cp_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
