#include<assert.h>
#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>
// #include<helper_cuda.h>
// #include<hepler_functions.h>


template<typename T>
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(T err, const char* const  func, const char* const  file, int const line)
{
    if(err != cudaSuccess)
    {
        std::cerr<<" Cuda runtime error at: " <<file<<": "<<line<<std::endl;
        std::cerr<<func<<":"<<cudaGetErrorString(err)<<std::endl;
        exit(0);
    }
}

#define  MEMORY_ALIGNMET 4096
#define ALIGN_UP(x, size) ((((size_t)x + (size - 1)) & (~(size - 1))))

bool bPinGenericMemory = false;

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAdd(float* a, float* b, float* c, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }

}

int main(int argc, char **argv)
{
    int nelem, deviceCount;
    int idev = 0;
    char* device = NULL;

    //unsigned int flags;

    size_t bytes;

    float *a, *b, *c;
    float *a_u, *b_u, * c_u;
    float *d_a, *d_b, *d_c;

    cudaDeviceProp deviceProp;

    // if(checkCmdLineFlag(argc, (const char **)argv), "help"){
    // printf("Usage:  simpleZeroCopy [OPTION]\n\n");
    // printf("Options:\n");
    // printf("  --device=[device #]  Specify the device to be used\n");
    // printf(
    //     "  --use_generic_memory (optional) use generic page-aligned for system "
    //     "memory\n");
    //     return EXIT_SUCCESS;
    // }

    cudaGetDeviceCount(&deviceCount);
    idev = atoi(device);

    if(idev > deviceCount || idev < 0)
    {
        fprintf(stderr, " Device number %d is invalid, will used default device 0. \n", idev);
        idev = 0;
    }

    // if(!checkCudaCapabilities(1,2))
    // {
    //     exit(EXIT_SUCCESS);
    // }

    bPinGenericMemory = true;
    CHECK_CUDA_ERROR(cudaSetDevice(idev));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp,idev));

    if(!deviceProp.canMapHostMemory)
    {
        fprintf(stderr, "Device %d does not support mapping CPU to host memory! \n", idev);
        exit(0);
    }

    nelem = 1048576;
    bytes = nelem * sizeof(float);

    if(bPinGenericMemory)
    {
        a_u = (float*)malloc(bytes + MEMORY_ALIGNMET);
        b_u = (float*)malloc(bytes + MEMORY_ALIGNMET);
        c_u = (float*)malloc(bytes + MEMORY_ALIGNMET);

        a = (float*)ALIGN_UP(a_u, MEMORY_ALIGNMET);
        b = (float*)ALIGN_UP(b_u, MEMORY_ALIGNMET);
        c = (float*)ALIGN_UP(c_u, MEMORY_ALIGNMET);

        cudaHostRegister(a, bytes, cudaHostRegisterMapped);
        cudaHostRegister(b, bytes, cudaHostRegisterMapped);
        cudaHostRegister(c, bytes, cudaHostRegisterMapped);

        // flags = cudaHostAllocMapped;
        // cudaHostAlloc((void**)&a, bytes, flags);
        // cudaHostAlloc((void**)&b, bytes, flags);
        // cudaHostAlloc((void**)&c, bytes, flags);

        cudaHostGetDevicePointer((void**)&d_a, (void*)a, 0);
        cudaHostGetDevicePointer((void**)&d_b, (void*)b, 0);
        cudaHostGetDevicePointer((void**)&d_c, (void*)c, 0);

        dim3 block_dim(256,1,1);
        dim3 grid_dim((nelem + block_dim.x - 1) / block_dim.x,1 , 1);
        vectorAdd<<<grid_dim,block_dim>>>(a, b, c, nelem);
        cudaDeviceSynchronize();

        cudaHostUnregister(a);
        cudaHostUnregister(b);
        cudaHostUnregister(c);
        free(a_u);
        free(b_u);
        free(c_u);

        // cudaFreeHost(a);
        // cudaFreeHost(b);
        // cudaFreeHost(c);
        return 0;

    }


}



