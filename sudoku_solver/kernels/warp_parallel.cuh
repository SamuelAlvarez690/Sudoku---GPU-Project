#pragma once
#include <cuda_runtime.h>

#define WARP_MAX_DEPTH 81
#define FULL_MASK 0xFFFFFFFFu

struct WarpState {
    int board[81];
    unsigned int row_mask[9];
    unsigned int col_mask[9];
    unsigned int box_mask[9];
};

__device__ __forceinline__ int wb(int r, int c) {
    return (r / 3) * 3 + (c / 3);
}

__device__ __forceinline__ void warp_min(int& val, int& pos) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        int v2 = __shfl_xor_sync(FULL_MASK, val, d);
        int p2 = __shfl_xor_sync(FULL_MASK, pos, d);

        if (v2 < val || (v2 == val && p2 < pos)) {
            val = v2;
            pos = p2;
        }
    }

    val = __shfl_sync(FULL_MASK, val, 0);
    pos = __shfl_sync(FULL_MASK, pos, 0);
}

__global__ void warp_sudoku_kernel(const int* __restrict__ d_puzzles, int* __restrict__ d_solutions, bool* __restrict__ d_solved, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;

    if (warp_id >= N) return;

    extern __shared__ WarpState smem[];
    WarpState& ws = smem[warp_in_block];

    ws.board[lane] = d_puzzles[warp_id * 81 + lane];
    ws.board[lane + 32] = d_puzzles[warp_id * 81 + lane + 32];
    if (lane < 17) {
        ws.board[lane + 64] = d_puzzles[warp_id * 81 + lane + 64];
    }
    __syncwarp(FULL_MASK);

    if (lane == 0) {
        for (int i = 0; i < 9; i++) {
            ws.row_mask[i] = 0u;
            ws.col_mask[i] = 0u;
            ws.box_mask[i] = 0u;
        }

        for (int i = 0; i < 81; i++) {
            int v = ws.board[i];
            if (v != 0) {
                int r = i / 9;
                int c = i % 9;
                int b = wb(r, c);
                unsigned int bit = 1u << (v - 1);

                ws.row_mask[r] |= bit;
                ws.col_mask[c] |= bit;
                ws.box_mask[b] |= bit;
            }
        }
    }
    __syncwarp(FULL_MASK);

    int stk_pos[WARP_MAX_DEPTH];
    unsigned int stk_tried[WARP_MAX_DEPTH];
    int stk_val[WARP_MAX_DEPTH];

    if (lane == 0) {
        for (int i = 0; i < WARP_MAX_DEPTH; i++) {
            stk_pos[i] = -1;
            stk_tried[i] = 0u;
            stk_val[i] = 0;
        }
    }
    __syncwarp(FULL_MASK);

    int  depth  = 0;
    bool solved = false;

    while (true) {
        int best_cnt = 10;
        int best_pos = 99;

        for (int k = 0; k < 3; k++) {
            int cell = lane + k * 32;
            if (cell >= 81) break;
            if (ws.board[cell] != 0) continue;

            int r = cell / 9;
            int c = cell % 9;
            int b = wb(r, c);

            unsigned int used = ws.row_mask[r] | ws.col_mask[c] | ws.box_mask[b];
            unsigned int cand = (~used) & 0x1FFu;
            int cnt = __popc(cand);

            if (cnt == 0) {
                best_cnt = 0;
                best_pos = cell;
                break;
            }

            if (cnt < best_cnt || (cnt == best_cnt && cell < best_pos)) {
                best_cnt = cnt;
                best_pos = cell;
            }
        }

        warp_min(best_cnt, best_pos);

        if (best_pos == 99) {
            solved = true;
            break;
        }

        while (best_cnt == 0) {
            if (depth == 0) goto done;

            depth--;

            if (lane == 0) {
                int cell = stk_pos[depth];
                int v = stk_val[depth];

                if (cell >= 0 && v != 0) {
                    int r = cell / 9;
                    int c = cell % 9;
                    int b = wb(r, c);
                    unsigned int bit = 1u << (v - 1);

                    ws.board[cell] = 0;
                    ws.row_mask[r] &= ~bit;
                    ws.col_mask[c] &= ~bit;
                    ws.box_mask[b] &= ~bit;
                }
            }
            __syncwarp(FULL_MASK);

            int retry_pos = __shfl_sync(FULL_MASK, (lane == 0) ? stk_pos[depth] : 0, 0);
            unsigned int tried = __shfl_sync(FULL_MASK, (lane == 0) ? stk_tried[depth] : 0u, 0);

            int r = retry_pos / 9;
            int c = retry_pos % 9;
            int b = wb(r, c);

            unsigned int used = ws.row_mask[r] | ws.col_mask[c] | ws.box_mask[b];
            unsigned int cand = (~used) & 0x1FFu & ~tried;

            best_pos = retry_pos;
            best_cnt = (cand == 0u) ? 0 : __popc(cand);
        }

        {
            unsigned int tried = __shfl_sync(FULL_MASK, (lane == 0 && stk_pos[depth] == best_pos) ? stk_tried[depth] : 0u, 0);

            int r = best_pos / 9;
            int c = best_pos % 9;
            int b = wb(r, c);

            unsigned int used = ws.row_mask[r] | ws.col_mask[c] | ws.box_mask[b];
            unsigned int cand = (~used) & 0x1FFu & ~tried;

            if (cand == 0u) {
                if (depth == 0) goto done;

                depth--;

                if (lane == 0) {
                    int cell = stk_pos[depth];
                    int v = stk_val[depth];

                    if (cell >= 0 && v != 0) {
                        int r2 = cell / 9;
                        int c2 = cell % 9;
                        int b2 = wb(r2, c2);
                        unsigned int bit2 = 1u << (v - 1);

                        ws.board[cell]  = 0;
                        ws.row_mask[r2] &= ~bit2;
                        ws.col_mask[c2] &= ~bit2;
                        ws.box_mask[b2] &= ~bit2;
                    }
                }
                __syncwarp(FULL_MASK);
                continue;
            }

            unsigned int bit = cand & (~cand + 1u);
            int digit = __ffs((int)bit);

            if (lane == 0) {
                if (stk_pos[depth] != best_pos) {
                    stk_pos[depth]   = best_pos;
                    stk_tried[depth] = 0u;
                    stk_val[depth]   = 0;
                }

                stk_tried[depth] |= bit;
                stk_val[depth] = digit;
                ws.board[best_pos] = digit;
                ws.row_mask[r] |= bit;
                ws.col_mask[c] |= bit;
                ws.box_mask[b] |= bit;
            }
            __syncwarp(FULL_MASK);

            depth++;

            if (lane == 0 && depth < WARP_MAX_DEPTH) {
                stk_pos[depth]   = -1;
                stk_tried[depth] = 0u;
                stk_val[depth]   = 0;
            }
            __syncwarp(FULL_MASK);
        }
    }

done:
    if (lane == 0) {
        d_solved[warp_id] = solved;
    }

    d_solutions[warp_id * 81 + lane] = ws.board[lane];
    d_solutions[warp_id * 81 + lane + 32] = ws.board[lane + 32];
    if (lane < 17) {
        d_solutions[warp_id * 81 + lane + 64] = ws.board[lane + 64];
    }
}

inline void launch_warp(const int* d_puzzles, int* d_solutions, bool* d_solved, int N, int tpb = 256)
{
    int total = N * 32;
    int blocks = (total + tpb - 1) / tpb;
    int wpb = tpb / 32;
    size_t smem = wpb * sizeof(WarpState);

    warp_sudoku_kernel<<<blocks, tpb, smem>>>(d_puzzles, d_solutions, d_solved, N);
}