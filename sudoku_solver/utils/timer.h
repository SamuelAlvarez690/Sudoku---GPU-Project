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
