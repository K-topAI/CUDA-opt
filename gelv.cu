#include<algorithm>
#include<cuda_bf16.h>
#include<cuda_fp16.h>
#include<cuda_fp8.h>
#include<cuda_runtime.h>
#include<float.h>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<math.h>
// #include<torch/extension.h>
// #include<torch/types.h>

#define WARP_SIZE 32
#define INT4(val)       (reinterpret_cast<int4*>(&val)[0])
#define FLOAT4(val)     (reinterpret_cast<float4*>(&val)[0])
#define HALF2(val)      (reinterpret_cast<half2*>(&val)[0])
#define BFLOAT2(val)    (reinterpret_cast<__nv_bfloat162*>(&val)[0])
#define LDST128BITS(val) (reinterpret_cast<float4*>(&val)[0])
#define MAX_EXP_F32     88.3762626647949f //指数的最大值，在f32时
#define MIN_EXP_F32     -88.3762626647949f
#define MAX_EXP_F16    __float2half(11.089866488461016f)
#define MIN_EXP_F16    __float2half(-9.704060527839234f)
#define SQRT_2_PI      M_SQRT2 * M_2_SQRTPI * 0.5f
#define HALF_1         __float2half(1.0f)
#define HALF_2         __float2half(2.0f)
#define HALF_DIV       __float2half(0.5f)

#define HALF_SQRT_2_PI  __float2half(M_SQRT2) * __float2half(M_2_SQRTPI)*HALF_DIV
#define HALF_V          __float2half(0.044715f)
#define HALF_GELU_OPS   gelu_tanh_approximate
#define GELU_OPS        gelu_tanh_approximate
__inline__ __device__ half gelu_tanh_approximate(half x)
{
    half x_cube = x* x * x;
    half temp = HALF_SQRT_2_PI *(x + HALF_V * x_cube);

    return  HALF_DIV * x *(HALF_1 + ((hexp(temp * HALF_2) - HALF_1) / (hexp(temp * HALF_2) + HALF_1)));
}

__inline__ __device__ float gelu_tanh_approximate(float x)
{
    return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f *x * x * x )));
}
__global__ void gelu_f32_kernel(float* x, float* y,int N)
{
    //最大处理量：blockDim.x * blockDim.y
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    if(idx < N)
    {
        float val = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
        y[idx] = GELU_OPS(val);
    }
}

__global__ void gelu_f32_kernel_B(float* x, float* y, int N)
{
    //最大处理量：blockDim.x * gridDim.x
    //支持网格级并行
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < N)
    {
        float val = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
        y[idx] = GELU_OPS(val);
    }
}


__global__ void gelu_f32x4_kernel(float* x, float *y, int N)
{
    
    int idx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    float4 x4 = FLOAT4(x[idx]);
    float4 y4{0.0f,0.0f,0.0f,0.0f};
    
    //clmap处理 
    x4.x =  fminf(fmaxf(x4.x, MIN_EXP_F32),MAX_EXP_F32);
    x4.y = fminf(fmaxf(x4.y, MIN_EXP_F32), MAX_EXP_F32);
    x4.z = fminf(fmaxf(x4.z, MIN_EXP_F32), MAX_EXP_F32);
    x4.w = fminf(fmaxf(x4.w, MIN_EXP_F32), MAX_EXP_F32);

    y4.x = GELU_OPS(x4.x);
    y4.y = GELU_OPS(x4.y);
    y4.z = GELU_OPS(x4.z);
    y4.w = GELU_OPS(x4.w);

    if(idx < N)
    {
        reinterpret_cast<float4*>(&y[idx])[0] = y4;
    }
}

//half 英伟达开发守则中要求使用half2 处理或者传递数据，因此不要在任何优化中实现half的版本
__global__ void gelu_f16x2_kernel(half* x, half* y, int N)
{
    int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
    half2 x2 = reinterpret_cast<half2*>(&x[idx])[0];
    half2 y2;
    x2.x = __hmin(__hmax(x2.x, MIN_EXP_F16), MAX_EXP_F16);
    x2.y = __hmin(__hmax(x2.x, MIN_EXP_F16), MAX_EXP_F16);

    y2.x = HALF_GELU_OPS(x2.x);
    y2.y = HALF_GELU_OPS(x2.y);
    
    if(idx < N)
    {
        reinterpret_cast<half2*>(&y[idx])[0] = y2;
    }
}

//128bit 一次数据处理的事务的大小
__global__ void gelu_f16x8_kernel(half* x, half* y, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    half x_pack[8];
    half y_pack[8];

    reinterpret_cast<float4*>(x_pack)[0] = reinterpret_cast<float4*>(&x[idx * 8])[0];
    #pragma unroll
    for(int i = 0; i < 8; i++)
    {
        half v = __hmin(__hmax(x_pack[i], MIN_EXP_F16), MAX_EXP_F16);
        y_pack[i] = HALF_GELU_OPS(v);
    }
    if(idx < N / 8)
    {
        reinterpret_cast<float4*>(&y[8 * idx])[0] = reinterpret_cast<float4*>(y_pack)[0];
    }

}