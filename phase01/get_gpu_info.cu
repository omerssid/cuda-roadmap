#include <iostream>
#include <cuda_runtime.h>
#include <vector>


using std::cout;
using std::cerr;
using std::endl;

int main()
{
    int deviceCount;
    /*
    cudaError_t cudaGetDeviceCount(int *count);
    Returns the number of compute-capable devices
    */
    cudaGetDeviceCount(&deviceCount);
    

    if (deviceCount == 0) {
        std::cout << "No CUDA-compatible GPU found!" << std::endl;
        return 1;
    } else 
    {
        cout << "Device Count: " << deviceCount << endl;
    }
    unsigned i = 0;
    std::vector<int> range(deviceCount);
    for (auto _ : range)
    {
        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, i);

        cout << "GPU " << i << ": " << prop.name << endl;
        cout << "  Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
        cout << "  Number of SMs: " << prop.multiProcessorCount << endl;
        cout << "  Warp Size: " << prop.warpSize << endl;
        cout << "  Max Grid Size: " << prop.maxGridSize[0] 
                << " x " << prop.maxGridSize[1] 
                << " x " << prop.maxGridSize[2] 
                << endl;
        cout << "  Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " 
                << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
        cout << "---------------------------------------" << endl;
    }


    return 0;
}