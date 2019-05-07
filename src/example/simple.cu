#include <stdio.h>
#include <stdlib.h>

// Device code
__global__ void VecAdd(int* A, int* B, int* C) {
    int i = blockDim.x * blockIdx.x * threadIdx.x;
    C[i] = A[i] + B[i];
}

// Host code
int main() {
    int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int N = 4096;
    size_t size = N * sizeof(int);

    // Allocate input vectors h_A and h_B in host memory (CPU)
    h_A = (int *) malloc(size);
    h_B = (int *) malloc(size);
    h_C = (int *) malloc(size);

    // Initialise h_A and h_B here
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate Vectors in device memory (GPU)
    cudaMalloc((void**) &d_A, size);
    cudaMalloc((void**) &d_B, size);
    cudaMalloc((void**) &d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = N / threadsPerBlock;

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}