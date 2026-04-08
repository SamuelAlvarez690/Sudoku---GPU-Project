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
            std::vector<char> h_ok(N);
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
