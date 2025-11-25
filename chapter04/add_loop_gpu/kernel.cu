// add_loop_gpu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define N 10

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x;
    printf("tid: %d\n", tid);
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N];
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

    // <<<b, t>>> b:�豸��ִ�к˺���ʱʹ�õĲ����߳̿�(Block)������t:CUDA Runtime��ÿ���߳̿��д������߳�����
    // N���߳̿� * 1���߳�/�߳̿� = N�������߳�
    // �������˺���ʱ�����ǽ������߳̿�(Block)������ָ��ΪN����������߳̿鼯��Ҳ��Ϊһ���̸߳�(Grid),
    // ���Ǹ���CUDA Runtime��������Ҫһ��һά���̸߳����а���N���߳̿顣
    add<<<N, 1 >>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return 0;
}