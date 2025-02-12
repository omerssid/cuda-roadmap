#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>


using std::cout;
using std::cerr;
using std::endl;

#define N 10000000


__global__ void vecAdd(float* A, float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}



int main ()
{
    // pre- allocate & initialize variable on host
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    // 0. allocate device memory (using cudaMalloc)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // 1. Copy data from host to device
    /*
    Copies data between host and device:
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    */
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Run Kernel
    /* 
    2.1 define kernel execution config
        This config determines how many threads and blocks are launched to 
        execute a kernel on the GPU. This is crucial for optimizing performance.
    */
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    /*
    kernel<<<numBlocks, numThreads>>>(args);
    where:
        numBlocks → Number of blocks per grid
        numThreads → Number of threads per block
    */
    auto start = std::chrono::high_resolution_clock::now();
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    // Ensure all device threads finish execution
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;

    // 3. Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 4. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    // Run CPU version for comparison
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end - start;

    // Print performance results
    std::cout << "GPU Time: " << gpu_time.count() * 1000 << " ms\n";
    std::cout << "CPU Time: " << cpu_time.count() * 1000 << " ms\n";

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C; 

    return 0;
}