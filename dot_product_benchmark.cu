#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common/book.h"

#define N (32 * 1024 * 1024) // 32M elements
#define THREADS_PER_BLOCK 256

// GPU kernel for vector dot product
// Using atomicAdd for final reduction (simplified approach)
// Note: A full parallel reduction is faster but more complex.
// For this demo, we'll do partial sums per block and then atomic add to global sum,
// or just simple parallel multiplication and then CPU reduction to keep it simple and focus on memory bandwidth.
//
// Let's implement: C[i] = A[i] * B[i], then we sum C on CPU or GPU.
// To purely test computation/bandwidth, we can just do vector addition or multiplication.
// Dot product requires reduction which adds synchronization overhead.
// User asked for Dot Product, so let's do:
// 1. GPU: Calculate partial products in parallel: temp[i] = a[i] * b[i]
// 2. GPU: (Optional) Parallel reduction (too complex for simple demo) -> Let's use a simpler approach:
//    Each thread calculates a[i]*b[i], and we can use atomicAdd to a global variable (slow due to contention).
//    Better approach for "simple demo":
//    Kernel computes partial dot product for its own grid-stride loop and atomicAdds to a block-shared variable, then block adds to global.

__global__ void dot_product_kernel(float *a, float *b, float *c, int n) {
    // Grid-stride loop approach
    // This allows the kernel to work even if N > GridDim * BlockDim
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0;
    for (int i = tid; i < n; i += stride) {
        sum += a[i] * b[i];
    }

    // Using atomicAdd to accumulate the result from each thread to a global variable
    // Note: Heavily contending on a single global address is slow. 
    // A better way is shared memory reduction, but let's start with this to see if it beats CPU.
    // Actually, atomicAdd to a single float from millions of threads is a huge bottleneck.
    // 
    // Optimization: Shared memory reduction per block
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = sum;
    
    __syncthreads();

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // Only thread 0 of each block adds to the global result
    if (cacheIndex == 0)
        atomicAdd(c, cache[0]);
}

void dot_product_cpu(float *a, float *b, float *c, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    *c = (float)sum;
}

int main() {
    float *a, *b, *c_cpu, *c_gpu;
    float *dev_a, *dev_b, *dev_c;

    // Allocate host memory
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c_cpu = (float*)malloc(sizeof(float));
    c_gpu = (float*)malloc(sizeof(float));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(float)));

    // Copy data to device
    clock_t start, end;
    double cpu_time_used, gpu_time_used;

    printf("Vector size: %d elements (%.2f MB per vector)\n", N, (float)N * sizeof(float) / 1024 / 1024);

    // --- CPU Computation ---
    start = clock();
    dot_product_cpu(a, b, c_cpu, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // ms
    printf("CPU Result: %.2f\n", *c_cpu);
    printf("CPU Time: %.2f ms\n", cpu_time_used);

    // --- GPU Computation ---
    
    // 1. Copy Host to Device
    start = clock();
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Reset result on GPU
    float zero = 0.0f;
    HANDLE_ERROR(cudaMemcpy(dev_c, &zero, sizeof(float), cudaMemcpyHostToDevice));

    // 2. Launch Kernel
    // Calculate grid size
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Limit grid size to max reasonable if N is huge (though for 32M it's fine)
    // Optimization: Use enough blocks to saturate GPU, but loop inside kernel handles the rest
    if (blocksPerGrid > 65535) blocksPerGrid = 65535; 

    dot_product_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    // 3. Copy result back
    HANDLE_ERROR(cudaMemcpy(c_gpu, dev_c, sizeof(float), cudaMemcpyDeviceToHost));
    
    end = clock();
    gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // ms

    printf("GPU Result: %.2f\n", *c_gpu);
    printf("GPU Time (Total including MemCpy): %.2f ms\n", gpu_time_used);
    
    // Measure Kernel Only Time
    start = clock();
    dot_product_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c, N);
    cudaDeviceSynchronize();
    end = clock();
    double kernel_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("GPU Time (Kernel Only): %.2f ms\n", kernel_time);


    // Cleanup
    free(a); free(b); free(c_cpu); free(c_gpu);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}

