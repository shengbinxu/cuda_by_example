#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common/book.h"

#define N (32 * 1024 * 1024) // 32M elements
#define THREADS_PER_BLOCK 256

// Kernel to initialize data directly on GPU
__global__ void init_data_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
}

// Dot product computation kernel (Same as before)
__global__ void dot_product_kernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0;
    for (int i = tid; i < n; i += stride) {
        sum += a[i] * b[i];
    }

    // Shared memory reduction per block
    __shared__ float cache[THREADS_PER_BLOCK];
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = sum;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

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
    // Host pointers (only for verification)
    float *a_host, *b_host, *c_cpu, *c_gpu;
    // Device pointers
    float *dev_a, *dev_b, *dev_c;

    // Allocate host memory (just for CPU verification)
    a_host = (float*)malloc(N * sizeof(float));
    b_host = (float*)malloc(N * sizeof(float));
    c_cpu = (float*)malloc(sizeof(float));
    c_gpu = (float*)malloc(sizeof(float));

    // Initialize host arrays for CPU calculation
    for(int i=0; i<N; i++) {
        a_host[i] = 1.0f;
        b_host[i] = 2.0f;
    }

    printf("Vector size: %d elements (%.2f MB per vector)\n", N, (float)N * sizeof(float) / 1024 / 1024);

    // --- CPU Computation ---
    clock_t start = clock();
    dot_product_cpu(a_host, b_host, c_cpu, N);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Result: %.2f\n", *c_cpu);
    printf("CPU Time: %.2f ms\n", cpu_time);

    // --- GPU Computation (Zero-Copy Init) ---
    
    // 1. Allocate Device Memory
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(float)));

    start = clock();

    // 2. Initialize Data on GPU (No Memcpy from Host!)
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocksPerGrid > 65535) blocksPerGrid = 65535;

    init_data_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dev_a, dev_b, N);
    HANDLE_ERROR(cudaGetLastError());
    
    // Reset result
    float zero = 0.0f;
    HANDLE_ERROR(cudaMemcpy(dev_c, &zero, sizeof(float), cudaMemcpyHostToDevice)); // Tiny copy (4 bytes)

    // 3. Compute
    dot_product_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c, N);
    HANDLE_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();

    // 4. Copy Result Back
    HANDLE_ERROR(cudaMemcpy(c_gpu, dev_c, sizeof(float), cudaMemcpyDeviceToHost)); // Tiny copy (4 bytes)

    end = clock();
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("GPU Result: %.2f\n", *c_gpu);
    printf("GPU Time (Total including Init on GPU): %.2f ms\n", gpu_time);
    
    // Compare
    printf("Speedup (GPU vs CPU): %.2fx\n", cpu_time / gpu_time);

    // Cleanup
    free(a_host); free(b_host); free(c_cpu); free(c_gpu);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}

