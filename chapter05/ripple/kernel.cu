// ripple
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr, int ticks) {
    // 映射坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = y * blockDim.x * gridDim.x + x;

    // 计算波纹
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

void save_to_ppm(unsigned char *ptr, int w, int h, int frame_idx) {
    char filename[64];
    sprintf(filename, "frame_%03d.ppm", frame_idx);
    
    FILE *fp = fopen(filename, "wb");
    // PPM header: P6 <width> <height> <maxval>
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int offset = (x + y * w) * 4; // RGBA from CUDA
            // Write RGB, skip A
            fwrite(ptr + offset, 1, 3, fp);
        }
    }
    fclose(fp);
}

int main() {
    int width = DIM;
    int height = DIM;
    size_t image_size = width * height * 4;

    // 1. Allocate Host Memory
    unsigned char *host_bitmap = (unsigned char*)malloc(image_size);

    // 2. Allocate Device Memory
    unsigned char *dev_bitmap;
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, image_size));

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    // 3. Loop to generate frames
    int num_frames = 100;
    printf("Generating %d frames...\n", num_frames);
    
    for(int ticks = 0; ticks < num_frames; ticks++) {
        // Kernel
        kernel<<<blocks, threads>>>(dev_bitmap, ticks);
        HANDLE_ERROR(cudaGetLastError());
        cudaDeviceSynchronize();

        // Copy back
        HANDLE_ERROR(cudaMemcpy(host_bitmap, dev_bitmap, image_size, cudaMemcpyDeviceToHost));

        // Save
        save_to_ppm(host_bitmap, width, height, ticks);
        
        if(ticks % 10 == 0) printf("."); 
        fflush(stdout);
    }
    printf("\nDone generating frames.\n");

    // 4. Cleanup
    HANDLE_ERROR(cudaFree(dev_bitmap));
    free(host_bitmap);

    return 0;
}
