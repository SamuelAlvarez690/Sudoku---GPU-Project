#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  COSC 4397 – GPU Sudoku Solver  –  self-extracting setup script
#  Run once on your server:   bash setup_sudoku.sh
#  Then:                       cd sudoku_solver && make run
# ─────────────────────────────────────────────────────────────────────────────
set -e

mkdir -p sudoku_solver/kernels sudoku_solver/utils
cd sudoku_solver

# ══════════════════════════════════════════════════════════════════════════════
#  Makefile
# ══════════════════════════════════════════════════════════════════════════════
cat > Makefile << 'MAKEFILE_EOF'
# ─── Sudoku Solver – COSC 4397 Final Project ────────────────────────────────
NVCC       := nvcc
CXX_STD    := --std=c++17
ARCH       := -arch=sm_86
OPT        := -O3 --use_fast_math
STACKSIZE  := -Xptxas -v
NVCCFLAGS  := $(CXX_STD) $(ARCH) $(OPT) $(STACKSIZE)
TARGET     := sudoku_bench
SRCS       := sudoku_bench.cu
INCLUDES   := -I.

.PHONY: all run clean info

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^
	@echo ""
	@echo "Build successful → ./$(TARGET)"

run: all
	@echo ""
	./$(TARGET)

info:
	@nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
	            --format=csv,noheader

clean:
	rm -f $(TARGET) *.o *.ptx
MAKEFILE_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  utils/timer.h
# ══════════════════════════════════════════════════════════════════════════════
cat > utils/timer.h << 'TIMER_EOF'
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ─── CUDA error checking ─────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── GPU event timer ─────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start); }
    float Stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ─── Warm-up helper ──────────────────────────────────────────────────────────
__global__ void warmup_kernel() { /* intentionally empty */ }
inline void gpu_warmup() {
    warmup_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}
TIMER_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  utils/puzzle_io.h
# ══════════════════════════════════════════════════════════════════════════════
cat > utils/puzzle_io.h << 'PUZZLE_EOF'
#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// A Sudoku board is stored as a flat array of 81 ints.
// 0 = empty cell, 1-9 = given clue.
using Board = int[81];

// Pretty-print a board to stdout
inline void print_board(const int* b) {
    for (int r = 0; r < 9; r++) {
        if (r % 3 == 0 && r != 0) printf("+-------+-------+-------+\n");
        for (int c = 0; c < 9; c++) {
            if (c % 3 == 0) printf("| ");
            int v = b[r * 9 + c];
            if (v == 0) printf(". ");
            else        printf("%d ", v);
        }
        printf("|\n");
    }
}

// Validate a completed board (all rows/cols/boxes use 1-9 exactly once)
inline bool validate_board(const int* b) {
    for (int r = 0; r < 9; r++) {
        int seen = 0;
        for (int c = 0; c < 9; c++) {
            int v = b[r*9+c];
            if (v < 1 || v > 9) return false;
            if (seen & (1 << v)) return false;
            seen |= (1 << v);
        }
    }
    for (int c = 0; c < 9; c++) {
        int seen = 0;
        for (int r = 0; r < 9; r++) {
            int v = b[r*9+c];
            if (seen & (1 << v)) return false;
            seen |= (1 << v);
        }
    }
    for (int br = 0; br < 3; br++) {
        for (int bc = 0; bc < 3; bc++) {
            int seen = 0;
            for (int dr = 0; dr < 3; dr++)
                for (int dc = 0; dc < 3; dc++) {
                    int v = b[(br*3+dr)*9 + (bc*3+dc)];
                    if (seen & (1 << v)) return false;
                    seen |= (1 << v);
                }
        }
    }
    return true;
}

// Built-in puzzle bank – Easy → Near-hardest
static const char* PUZZLE_BANK[] = {
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
    "800000000003600000070090200060005030000403001008070600002006004500008000010000006",
    "000000012000035000000600070700000300000400800100000000000120000080000040050000600",
    "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000001",
};
static const char* PUZZLE_NAMES[] = {
    "Easy", "Medium", "Hard", "Expert", "Extreme", "Near-hardest"
};
static const int NUM_PUZZLES = 6;

inline void load_puzzle(const char* str, int* board) {
    for (int i = 0; i < 81; i++) {
        char ch = str[i];
        board[i] = (ch >= '1' && ch <= '9') ? (ch - '0') : 0;
    }
}

inline std::vector<std::vector<int>> generate_puzzle_batch(int N) {
    std::vector<std::vector<int>> batch(N, std::vector<int>(81));
    for (int i = 0; i < N; i++)
        load_puzzle(PUZZLE_BANK[i % NUM_PUZZLES], batch[i].data());
    return batch;
}
PUZZLE_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  kernels/naive.cuh
# ══════════════════════════════════════════════════════════════════════════════
cat > kernels/naive.cuh << 'NAIVE_EOF'
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

__global__ void naive_sudoku_kernel(const int* __restrict__ d_puzzles,
                                    int*       __restrict__ d_solutions,
                                    bool*      __restrict__ d_solved,
                                    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int board[81];
    #pragma unroll
    for (int i = 0; i < 81; i++) board[i] = d_puzzles[tid * 81 + i];

    d_solved[tid] = naive_solve(board);

    #pragma unroll
    for (int i = 0; i < 81; i++) d_solutions[tid * 81 + i] = board[i];
}

inline void launch_naive(const int* d_puzzles, int* d_solutions,
                         bool* d_solved, int N, int tpb = 64)
{
    naive_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
NAIVE_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  kernels/bitboard.cuh  (stub)
# ══════════════════════════════════════════════════════════════════════════════
cat > kernels/bitboard.cuh << 'BB_EOF'
#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Bitboard Sudoku Solver  [TODO]
//  Strategy: encode row/col/box constraints as 9-bit bitmasks.
//  candidate(cell) = ~(row_mask[r] | col_mask[c] | box_mask[b]) & 0x1FF
//  Validity check = single AND vs. 27 array reads in naive.
// ============================================================

__global__ void bitboard_sudoku_kernel(const int* __restrict__ d_puzzles,
                                       int*       __restrict__ d_solutions,
                                       bool*      __restrict__ d_solved,
                                       int N)
{ /* TODO */ }

inline void launch_bitboard(const int* d_puzzles, int* d_solutions,
                             bool* d_solved, int N, int tpb = 128)
{
    bitboard_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
BB_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  kernels/constraint_prop.cuh  (stub)
# ══════════════════════════════════════════════════════════════════════════════
cat > kernels/constraint_prop.cuh << 'CP_EOF'
#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Constraint Propagation Solver  [TODO]
//  Strategy: run naked-singles pass before backtracking.
//  Cells with only one candidate are filled immediately.
//  Easy/medium puzzles may solve with zero backtracking.
// ============================================================

__global__ void cp_sudoku_kernel(const int* __restrict__ d_puzzles,
                                 int*       __restrict__ d_solutions,
                                 bool*      __restrict__ d_solved,
                                 int N)
{ /* TODO */ }

inline void launch_cp(const int* d_puzzles, int* d_solutions,
                      bool* d_solved, int N, int tpb = 64)
{
    cp_sudoku_kernel<<<(N+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
CP_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  kernels/warp_parallel.cuh  (stub)
# ══════════════════════════════════════════════════════════════════════════════
cat > kernels/warp_parallel.cuh << 'WP_EOF'
#pragma once
#include <cuda_runtime.h>

// ============================================================
//  Warp-Parallel Sudoku Solver  [TODO]
//  Strategy: assign 1 warp (32 threads) per puzzle.
//  Threads cooperate on constraint checking via warp shuffles.
//  Key intrinsics: __ballot_sync, __any_sync, __shfl_sync
// ============================================================

__global__ void warp_sudoku_kernel(const int* __restrict__ d_puzzles,
                                   int*       __restrict__ d_solutions,
                                   bool*      __restrict__ d_solved,
                                   int N)
{
    // warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32
    // lane_id = threadIdx.x % 32
    /* TODO */
}

inline void launch_warp(const int* d_puzzles, int* d_solutions,
                        bool* d_solved, int N, int tpb = 256)
{
    // 32 threads per puzzle
    int total = N * 32;
    warp_sudoku_kernel<<<(total+tpb-1)/tpb, tpb>>>(d_puzzles, d_solutions, d_solved, N);
}
WP_EOF

# ══════════════════════════════════════════════════════════════════════════════
#  sudoku_bench.cu  – main benchmark driver
# ══════════════════════════════════════════════════════════════════════════════
cat > sudoku_bench.cu << 'BENCH_EOF'
// sudoku_bench.cu  – COSC 4397 Final Project benchmark driver
//
// Output format mirrors the GEMM homework:
//   =================================================================
//   Sudoku solver  --  batch size: 256 puzzles
//   -----------------------------------------------------------------
//   Kernel                  Time (ms)    Puzzles/s    Solved
//   -----------------------------------------------------------------
//   Naive                      12.34       5187.1    256/256
//   ...
//   =================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "utils/timer.h"
#include "utils/puzzle_io.h"
#include "kernels/naive.cuh"
// Uncomment as you implement each kernel:
// #include "kernels/bitboard.cuh"
// #include "kernels/constraint_prop.cuh"
// #include "kernels/warp_parallel.cuh"

// ─── Benchmark config ────────────────────────────────────────────────────────
static const int BATCH_SIZES[] = {64, 256, 1024};
static const int NUM_BATCHES   = 3;
static const int WARMUP_REPS   = 2;
static const int TIMING_REPS   = 5;

// ─── Kernel registry ─────────────────────────────────────────────────────────
struct KernelEntry {
    const char* name;
    void (*launch)(const int*, int*, bool*, int, int);
    int  tpb;   // threads per block
};

static KernelEntry KERNELS[] = {
    { "Naive",           launch_naive,    64  },
    // { "Bitboard",        launch_bitboard, 128 },
    // { "Constraint Prop", launch_cp,        64 },
    // { "Warp Parallel",   launch_warp,     256 },
};
static const int NUM_KERNELS = (int)(sizeof(KERNELS) / sizeof(KERNELS[0]));

// ─── CPU reference (correctness only) ───────────────────────────────────────
static bool cpu_is_valid(const int* b, int pos, int val) {
    int r=pos/9, c=pos%9, br=(r/3)*3, bc=(c/3)*3;
    for (int i=0;i<9;i++) {
        if (b[r*9+i]==val||b[i*9+c]==val) return false;
        if (b[(br+i/3)*9+(bc+i%3)]==val) return false;
    }
    return true;
}
static bool cpu_solve(int* b) {
    int pos=-1;
    for (int i=0;i<81;i++) if (b[i]==0){pos=i;break;}
    if (pos==-1) return true;
    for (int v=1;v<=9;v++) {
        if (cpu_is_valid(b,pos,v)) {
            b[pos]=v;
            if (cpu_solve(b)) return true;
            b[pos]=0;
        }
    }
    return false;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
static void sep(char c, int w=65){for(int i=0;i<w;i++)putchar(c);putchar('\n');}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s\n\n", prop.name);

    // Increase device stack for deep recursion in backtracker
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 65536));

    gpu_warmup();

    for (int bi = 0; bi < NUM_BATCHES; bi++) {
        int N = BATCH_SIZES[bi];

        // Build flat puzzle array
        auto batch = generate_puzzle_batch(N);
        std::vector<int> h_puzzles(N * 81);
        for (int i = 0; i < N; i++)
            memcpy(h_puzzles.data() + i*81, batch[i].data(), 81*sizeof(int));

        // GPU buffers
        int  *d_puzzles, *d_solutions;
        bool *d_solved;
        CUDA_CHECK(cudaMalloc(&d_puzzles,   N*81*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_solutions, N*81*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_solved,    N*sizeof(bool)));

        // Header
        sep('=');
        printf("Sudoku solver  --  batch size: %d puzzle%s\n", N, N>1?"s":"");
        sep('-');
        printf("%-24s  %10s  %12s  %9s\n",
               "Kernel", "Time (ms)", "Puzzles/s", "Solved");
        sep('-');

        for (int ki = 0; ki < NUM_KERNELS; ki++) {
            auto& K = KERNELS[ki];

            // Warmup
            for (int r = 0; r < WARMUP_REPS; r++) {
                CUDA_CHECK(cudaMemcpy(d_puzzles, h_puzzles.data(),
                                     N*81*sizeof(int), cudaMemcpyHostToDevice));
                K.launch(d_puzzles, d_solutions, d_solved, N, K.tpb);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Timed runs
            float total_ms = 0.f;
            GpuTimer timer;
            for (int r = 0; r < TIMING_REPS; r++) {
                CUDA_CHECK(cudaMemcpy(d_puzzles, h_puzzles.data(),
                                     N*81*sizeof(int), cudaMemcpyHostToDevice));
                timer.Start();
                K.launch(d_puzzles, d_solutions, d_solved, N, K.tpb);
                total_ms += timer.Stop();
            }
            float avg_ms = total_ms / TIMING_REPS;

            // Copy back
            std::vector<int>  h_sol(N*81);
            std::vector<bool> h_ok(N);
            CUDA_CHECK(cudaMemcpy(h_sol.data(), d_solutions, N*81*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ok.data(),  d_solved,    N*sizeof(bool),   cudaMemcpyDeviceToHost));

            // Correctness
            int n_ok = 0;
            for (int i = 0; i < N; i++)
                if (h_ok[i] && validate_board(h_sol.data()+i*81)) n_ok++;

            float pps = (float)N / (avg_ms * 1e-3f);
            printf("%-24s  %10.2f  %12.1f  %4d/%-4d",
                   K.name, avg_ms, pps, n_ok, N);
            if (n_ok < N) printf("  *** MISMATCH ***");
            printf("\n");
        }

        sep('=');
        printf("\n");

        CUDA_CHECK(cudaFree(d_puzzles));
        CUDA_CHECK(cudaFree(d_solutions));
        CUDA_CHECK(cudaFree(d_solved));
    }
    return 0;
}
BENCH_EOF

echo ""
echo "✓  Project created in:  $(pwd)"
echo ""
echo "Files:"
find . -type f | sort
echo ""
echo "─────────────────────────────────────────"
echo "  Next steps:"
echo "    make          ← compile"
echo "    make run      ← compile + run"
echo "    make clean    ← remove binary"
echo "    make info     ← show GPU"
echo "─────────────────────────────────────────"
