#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../../common/book.h"

#define DIM 1024
#define NUM_FRAMES 100

// --- GPU KERNEL ---
__global__ void kernel_gpu(unsigned char* ptr, int ticks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * blockDim.x * gridDim.x + x;

    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d / 10.0f - ticks / 7.0f) /
                                         (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

// --- CPU FUNCTION ---
void kernel_cpu(unsigned char* ptr, int ticks) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = y * DIM + x;

            float fx = x - DIM / 2;
            float fy = y - DIM / 2;
            float d = sqrtf(fx * fx + fy * fy);
            unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                                 cos(d / 10.0f - ticks / 7.0f) /
                                                 (d / 10.0f + 1.0f));
            ptr[offset * 4 + 0] = grey;
            ptr[offset * 4 + 1] = grey;
            ptr[offset * 4 + 2] = grey;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

int main() {
    size_t image_size = DIM * DIM * 4;
    unsigned char *host_bitmap = (unsigned char*)malloc(image_size);
    unsigned char *dev_bitmap;
    
    printf("Generating %d frames at %dx%d resolution...\n\n", NUM_FRAMES, DIM, DIM);

    // ---------------------------
    // 1. Measure CPU Performance
    // ---------------------------
    printf("Running CPU version...\n");
    clock_t start_cpu = clock();
    
    for(int i=0; i<NUM_FRAMES; i++) {
        kernel_cpu(host_bitmap, i);
    }
    
    clock_t end_cpu = clock();
    double cpu_time_ms = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time: %.2f ms (%.2f FPS)\n", cpu_time_ms, NUM_FRAMES / (cpu_time_ms / 1000.0));


    // ---------------------------
    // 2. Measure GPU Performance
    // ---------------------------
    printf("\nRunning GPU version...\n");
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, image_size));
    
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    // Warmup
    kernel_gpu<<<blocks, threads>>>(dev_bitmap, 0);
    cudaDeviceSynchronize();

    clock_t start_gpu = clock();

    for(int i=0; i<NUM_FRAMES; i++) {
        kernel_gpu<<<blocks, threads>>>(dev_bitmap, i);
        // Note: In a real rendering pipeline, we might not copy back every frame 
        // if we display directly from GPU. But for fair comparison (generating data),
        // we usually include calculation time. 
        // If we include Memcpy, it tests PCIe bandwidth too.
        // Let's include Memcpy to simulate "generating and getting the result".
        HANDLE_ERROR(cudaMemcpy(host_bitmap, dev_bitmap, image_size, cudaMemcpyDeviceToHost));
    }
    // Ensure GPU is finished
    cudaDeviceSynchronize();

    clock_t end_gpu = clock();
    double gpu_time_ms = ((double)(end_gpu - start_gpu)) / CLOCKS_PER_SEC * 1000.0;
    printf("GPU Time: %.2f ms (%.2f FPS)\n", gpu_time_ms, NUM_FRAMES / (gpu_time_ms / 1000.0));

    // ---------------------------
    // Results
    // ---------------------------
    printf("\nSpeedup: %.2fx\n", cpu_time_ms / gpu_time_ms);

    // Cleanup
    free(host_bitmap);
    cudaFree(dev_bitmap);
    return 0;
}

