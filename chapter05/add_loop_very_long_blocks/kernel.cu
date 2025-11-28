// add_loop_very_long_blocks

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define N (65536*1024)

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid: %d, blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d, gridDim: %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x, gridDim.x);
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        // gridDim：代表 Grid 的尺寸。也就是“这一批任务总共有多少个 Block”。
        // blockDim：代表 Block 的尺寸。也就是“每个 Block 总共有多少个 Thread”。
        /**
         */
        tid +=  gridDim.x * blockDim.x;  // ������������Thread������ = ������Block����(gridDim.x) * ÿ��Block������Thread������(blockDim.x)
                                         // ��ÿ��Thread���� N / (gridDim.x * blockDim.x)������
    }
}

int main() {
    int* a, * b, * c;
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));

    int* dev_a, * dev_b, * dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    // ����128��Block��ÿ��Block����128���߳�
    // ��ÿ��Thread���� N / 16384������
    add<<<128, 128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    free(a);
    free(b);
    free(c);

    return 0;
}