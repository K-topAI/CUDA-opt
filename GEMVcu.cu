#include<cuda.h>
#include<cuda_bf16.h>
#include<cuda_fp8.h>
#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<vector>

#define WARP_SIZE 32

__device__ __forceinline__  float warp_sum_f32(float val)
{
    #pragma unroll
    for(int mask = 16 ; mask >=1; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff,val,mask,32);
    }
    return val;
}

__global__ void sgemv_kernel_f32(float* a, float* v, float* y, int w ,int h)
{
    //一个warp 处理一行数据
    //矩阵： a, 向量: b, y = a * b
    int row = threadIdx.y + blockIdx.x * blockDim.y;
    int tx = threadIdx.x;
    //int warp_id = tx / WARP_SIZE;
    int lane_id = tx % WARP_SIZE;

    if(row < h)
    {
        float sum = 0.0f;
        int num_warps = (w + WARP_SIZE -1) / WARP_SIZE;

        for(int i = 0; i < num_warps; ++i)
        {
            int warp_row = i * WARP_SIZE + lane_id;
            sum += a[row * w + warp_row] * v[warp_row];
        }
        for(int mask = 16; mask >=1; mask >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffff,sum,mask);
        }
        if(lane_id == 0)
        {
            y[row] = sum;
        }
    }
} 

__global__ void gemv_kernel_f32x4(float* __restrict__  a, float* __restrict__ v, float *y, int w ,int h)
{
    int row = threadIdx.y + blockIdx.y * blockDim.x;
    int tx = threadIdx.x;
    int lane_id = tx %  WARP_SIZE;

    if(row < h)
    {
        float sum = 0.0f;
        int num_warps = (w + WARP_SIZE - 1) / WARP_SIZE;
        for(int i =0; i < num_warps / 4; i++)
        {
            int warp_row_idx = i * WARP_SIZE + lane_id;
            float4 a1 = reinterpret_cast<float4*>(&a[row * w + warp_row_idx *4])[0];
            float4 b1 = reinterpret_cast<float4*>(&v[warp_row_idx *4])[0];

            sum += a1.x * b1.x;
            sum += a1.y * b1.y;
            sum += a1.z * b1.z;
            sum += a1.w * b1.w;
        }
        for(int mask = 16; mask >= 1; mask >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffff,sum,mask);
        }
        if(lane_id == 0)
        {
            y[row] = sum;
        }

    }

}

__global__ void gemv_kernel_2xwarp(float* a, float* v, float* y, int w, int h)
{
    int row = threadIdx.y + blockIdx.y * blockDim.x;
    int tx = threadIdx.x;
    int lane_id = tx % WARP_SIZE;
    int lane_id_2 = lane_id % 2;
    int num_col = lane_id / (WARP_SIZE / 2);
    if(row < h)
    {
        int g_row_idx = row *2 + lane_id_2;
        float sum = 0.0f;
        
        sum = a[g_row_idx * w + g_row_idx] * v[g_row_idx];

        for(int mask = 16; mask >= 1; mask >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffff,sum,mask);
        }
        if(num_col == 0)
        {
            y[g_row_idx] = sum;
        }
    }

}