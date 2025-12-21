#include<cassert>
#include<vector>
#include<algorithm>
#include<cuda_fp16.h>
#include<cuda_bf16.h>
#include<cuda_runtime.h>
#include<float.h>
#include<mma.h>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <utility>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val),#val,__FILE__,__LINE__)
template<class T>
void check(T err,const char* const func,const char* const file, int const line )
{
    if(err!=cudaSuccess)
    {
        std::cerr <<"cuda runtime Error at: "<<file<<":"
                                                <<line<<std::endl;
        std::cerr<<cudaGetErrorString(err)<<" "<<func <<std::endl;
        std::exit(0);
    }
}

#define CHECK_LAST_ERROR() check_last(__FILE__,__LINE__)
void check_last(const char * const file,int const line)
{
    cudaError_t const  err{cudaGetLastError()};
    if(err != cudaSuccess)
    {
        std::cerr<<"cuda runtime error at: " << file << " " << 
                                                line << std::endl;
        std::cerr<<cudaGetErrorString(err)<<std::endl;
        std::exit(0);
    }
}

__global__ void mat_trans_smem_naive_kernel(const int *dev,const int M, int const N
                                            , int* t_dev)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int s_data[32][32];

    if(row < M && col < N)
    {
        s_data[threadIdx.x][threadIdx.y] = dev[col + row * N];
        __syncthreads();

        int t_row = blockDim.x *blockIdx.x + threadIdx.x;
        int t_col = blockDim.y *blockIdx.y + threadIdx.y;
        if(t_row < N && t_col < M)
        {
            t_dev[t_row *M + t_col] = s_data[threadIdx.y][threadIdx.x];
        }
    }
}
__global__ void mat_trans_smem_swizzle_kernel(const int *dev,const int M, const int N
                                            ,int *t_dev)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int s_data[32][32];
    if(row < M && col< N)
    {
        s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev[col + row * N];
        __syncthreads();

        int t_row = blockDim.x * blockIdx.x + threadIdx.x;
        int t_col = blockDim.y * blockIdx.y + threadIdx.y;

       if(t_row < N &&t_col <M)
       {
            t_dev[t_col + t_row * M] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
       }

    }
}
int main(int argc,char *argv[])
{
    int M = 1024;
    int N = 1024;
    if(argc > 1)
    {
        M = std::stoi(argv[1]);
    }

    if(argc > 2)
    {
        N = std::stoi(argv[2]);
    }

    //size_t matrix_size = M * N * sizeof(int);

    int *dev;
    int *t_dev;
    cudaMalloc(&dev, M * N * sizeof(int));
    CHECK_CUDA_ERROR(cudaMalloc(&t_dev, M * N * sizeof(int)));

    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

     dim3 block(32,32);
     dim3 grid(N / 32, N / 32);

    mat_trans_smem_naive_kernel<<<grid,block>>>(dev,M,N,t_dev);
    cudaDeviceSynchronize();
    mat_trans_smem_swizzle_kernel<<<grid,block>>>(dev,M,N,t_dev);
    cudaDeviceSynchronize();

    printf("Done \n");
     cudaFree(dev);
     cudaFree(t_dev);

    return 1;
}


