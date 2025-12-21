#include<cuda.h>
#include<cuda_runtime.h>
//#include <torch/extension.h>
#include<random>
#include<chrono>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <utility>
#include <iostream>
#include <vector_types.h> 

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 32 
#endif

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif

constexpr int URF{UNROLL_FACTOR};

#ifndef SOFTMAX_VARIANT
#define SOFTMAX_VARIANT 8
#endif

#ifndef WIDTH
#define WIDTH 0
#endif

#define DIV_CEIL(X,Y) (((X) + (Y) - 1) / (Y))
#define WARP_SIZE 32
template<typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{//element_wise 
     int row = threadIdx.y + blockIdx.y * blockDim.y;
     int col = threadIdx.x + blockIdx.x * blockDim.x;

    if( row < h && col < w)
    {
        float maxval = 0.0f;
        for(int i = 0; i < w; i++ )
        {
            maxval = fmaxf(maxval, a[row * w + i]);
        }

        float divisor = 0.0f;
        for(int i = 0; i < w; i++)
        {
            divisor += __expf(a[row * w + i] - maxval);
        }
        b[row * w + col] = __expf(a[row * w + col] - maxval) / divisor;
    }
}

template<typename scalar_t>
__global__ void softmax_kernel_by_row_wise(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
    constexpr int row = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int ty = threadIdx.y;

    __shared__  float reduction[BLOCK_DIM_Y];

    if(row < h)
    {
        float maxval = 0.0f;
        for(int i = ty * BLOCK_DIM_Y; i < min(w, (ty+1) * BLOCK_DIM_Y); ++i)
        {
            maxval = fmaxf(maxval, a[row * w + i]);

        }
        reduction[ty] = maxval;
        for(int stride = BLOCK_DIM_Y >> 1; stride >= 1; stride >>= 1)
        {
            __syncthreads();
            if(ty < stride)
            {
                reduction[ty] = fmaxf(reduction[ty], reduction[ty + stride]);
            }
        }

       __syncthreads();
       maxval = reduction[0];
       float divisor = 0.0f;
       for(int i = ty * BLOCK_DIM_Y; i < min(w, (ty +1) * BLOCK_DIM_Y); ++i)
       {
            divisor += __expf(a[row * w + i] - maxval);
       }
       reduction[ty] =divisor;
       for(int stride = BLOCK_DIM_Y >> 1; stride >= 1; stride >>= 1)
       {
        __syncthreads();
        if(ty < stride)
        {
            reduction[ty] += reduction[ty + stride];
        }
       }
       __syncthreads();
       divisor = reduction[0];

       for(int i = ty; i < w; i += BLOCK_DIM_Y)
       {
         b[row * w + i] = __expf(a[row * w + i]- maxval) / divisor;
       }
    }

}

template<typename scalar_t>
__global__ void softmax_kernel_by_skip_stirde(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
    //相比于row_wise每个线程连续访问，修改为每个线程不连续访问，从而触发合并事务访问，和提高并行度
    constexpr int row = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int ty = threadIdx.y;

    __shared__ float reduction[BLOCK_DIM_Y];
    if(row < h)
    {
        float maxval = 0.0f;
        for(int i = ty; i < w; i += BLOCK_DIM_Y)
        {
            maxval = fmaxf(maxval, a[row * w + i]);
        }
        reduction[ty] = maxval;
        for(int stride = BLOCK_DIM_Y / 2; stride >= 1; stride >>= 1)
        {
            __syncthreads();
            if(ty < stride)
            {
                reduction[ty] = fmaxf(reduction[ty],reduction[ty + stride]);
            }
        }
        __syncthreads();
        maxval = reduction[0];

        float divisor = 0.0f;
        for(int i = ty; i < w; i+=BLOCK_DIM_Y)
        {
            divisor += __expf(a[ row * w + i] - maxval);
        }
        reduction[ty] = divisor;
        for(int stride = BLOCK_DIM_Y / 2; stride >= 1; stride >>= 1)
        {
            __syncthreads();
            if(ty < stride)
            {
                reduction[ty] += reduction[ty + stride];
            }
        }
        __syncthreads();
        divisor = reduction[0];

        for(int i = ty; i < w; i+=BLOCK_DIM_Y)
        {
            b[row * w + i] = __expf(a[row * w + i] - maxval) / divisor;
        }


    }
}

template<typename scalar_t>
__global__ void softmax_kernel_by_skip_stride_warp(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w , int h )
{
    constexpr int row = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int ty = threadIdx.y;
    constexpr int warp_id = ty / WARP_SIZE;
    constexpr int lane_id = ty % WARP_SIZE;
    
    __shared__ float reduction[BLOCK_DIM_Y / WARP_SIZE];
    if(row < h)
    {
        float maxval = 0.0f;
        for(int i = ty; i < w; i+=BLOCK_DIM_Y)
        {
            maxval = fmaxf(maxval, a[row * w + i]);
        }
        for(int mask = 16; mask >= 1; mask >>=1)
        {
            //warp 内规约
            maxval = fmaxf(maxval,__shfl_xor_sync(0xffffffff, maxval, mask, 32));
        }
        if(lane_id == 0)
        {
            //每个warp中的第一个写入warp内的最大值
            reduction[warp_id] = maxval;
        }
        __syncthreads();

        if(warp_id == 0)
        {   
            // 第一个warp归约所有warp的结果
            maxval = ty < BLOCK_DIM_Y / 32 ? reduction[ty] : 0;
            for(int mask = 16; mask >= 1; mask >>= 1)
            {
                maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
            } 
        }

        if(ty == 0)
        {
            //第一个warp归约所有warp的结果，由第一个线程写入最大值
            reduction[ty] = maxval;
        }
        __syncthreads();

        maxval = reduction[0];

        float divisor = 0.0f;
        for(int i = ty; i < w; i += BLOCK_DIM_Y)
        {
            divisor += __expf(a[row * w + i] -maxval);
        }
        for(int mask = 16; mask >= 1; mask>>=1)
        {
            divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32); 
        }
        if(lane_id == 0)
        {
            reduction[warp_id] = divisor;
        }
        __syncthreads();

        if(warp_id == 0)
        {
            divisor = ty < BLOCK_DIM_Y / 32 ? reduction[32] : 0;
            for(int mask = 16; mask >= 1; mask >>=1)
            {
                divisor+= __shfl_xor_sync(0xffffffff, divisor, mask, 32);
            }
        }

        if(ty == 0)
        {
            reduction[ty] = divisor;
        }
        __syncthreads();
        divisor = reduction[0];

        for(int i = ty; i < w; i+=BLOCK_DIM_Y)
        {
            b[row * w + i] = __expf(a[row * w + i] - maxval) /divisor;
        }
    }
}

template<typename scalar_t>
__global__ void softmax_kernel_by_coalescedmemory(scalar_t* __restrict__ a, scalar_t* __restrict__ b,int w, int h)
{
    constexpr int row = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int ty =threadIdx.y;
    __shared__ float reduction[BLOCK_DIM_Y / 2];//参与规约的共享内存减半

    if(row < h)
    {
        float maxval = 0.0f;
        for(int i = ty; i < w / 4; i+=BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(&a[row * w + i *4])[0];
            maxval = fmaxf(maxval, val.x);
            maxval = fmaxf(maxval, val.y);
            maxval = fmaxf(maxval, val.z);
            maxval = fmaxf(maxval, val.w); 
        }

        if(ty >= BLOCK_DIM_Y /2)
        {   
            //只存储后一半的最大值；前一半的最大值使用寄存器中的临时值
            reduction[ty - BLOCK_DIM_Y / 2] = maxval;
        }
         __syncthreads();
         //树状归约
        for(int stride = BLOCK_DIM_Y /2; stride >= 1; stride >>= 1)
        {
           
            if(ty < stride)
            {
                maxval = fmaxf(maxval, reduction[ty]);
                if(ty >= stride / 2)
                {
                    //只存储后一半的最大值；前一半的最大值使用寄存器中的临时值
                    reduction[ty - stride /2] = maxval;
                }
            }

        }

        __syncthreads();
        maxval = reduction[0];

        float divisor = 0.0f;
        for(int i = ty; i < w / 4; i+=BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(a[row * w + i * 4])[0];
            divisor += __expf(val.x -maxval);
            divisor += __expf(val.y -maxval);
            divisor += __expf(val.z -maxval);
            divisor += __expf(val.w -maxval);

            if(ty >= BLOCK_DIM_Y / 2)
            {
                reduction[ty - BLOCK_DIM_Y / 2] = divisor;
            }
        }
        __syncthreads();

        for(int stride = BLOCK_DIM_Y / 2; stride >= 1; stride >>=1)
        {
            if(ty < stride)
            {
                divisor+= reduction[ty];
                if(ty >= stride / 2)
                {
                    reduction[ty - stride / 2] = divisor;
                }
            }
        }
        __syncthreads();

        divisor = reduction[0];

        for(int i = ty; i < w / 4; i+=BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(&a[row * w + 4 * i])[0];
            
            val.x = __expf(val.x - maxval) / divisor;
            val.y = __expf(val.y - maxval) / divisor;
            val.z = __expf(val.z - maxval) / divisor;
            val.w = __expf(val.w - maxval) / divisor;

            reinterpret_cast<float4*>(&b[row * w + 4 * i])[0] = val;
        }


    }
}

template<typename scalar_t>
__global__ void softmax_kernel_by_coalsced_warp(scalar_t * a, scalar_t* b, int w, int h)
{
    constexpr int row = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int ty  = threadIdx.y;
    constexpr int warp_id = ty / 32;
    constexpr int lane_id =ty % 32;
    __shared__ float reduction[BLOCK_DIM_Y / 32];
   
   if(row < h)
   {
        float maxval = 0.0f;
        for(int  i = ty; i < w / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(a[row * w + i * 4])[0];
            maxval = fmaxf(maxval,val.x);
            maxval = fmaxf(maxval,val.y);
            maxval = fmaxf(maxval,val.z);
            maxval = fmaxf(maxval,val.w);
        }
        #pragma unroll
        for(int mask = 16; mask >= 1; mask >>= 1)
        {
            maxval = fmaxf(maxval,__shfl_xor_sync(0xffffffff,maxval,mask,32));
        }
        if(lane_id == 0)
        {
            reduction[warp_id] = maxval;
        }
        __syncthreads();

        if(warp_id == 0)
        {
            maxval =  ty < BLOCK_DIM_Y / 32 ? reduction[ty] : 0;
            #pragma unroll
            for(int mask = 16; mask >= 1; mask >>= 1)
            {
                maxval = fmaxf(maxval,__shfl_xor_sync(0xffffffff,maxval,mask,32));
            }
        }
        if(ty == 0)
        {
            reduction[0] = maxval;
        }
        __syncthreads();
        maxval =reduction[0];

        float divisor =0.0f;
        for(int i= ty; i < w / 4; i+=BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(a[row * w + i * 4])[0];
            divisor += __expf(val.x - maxval);
            divisor += __expf(val.y - maxval);
            divisor += __expf(val.z - maxval);
            divisor += __expf(val.w - maxval);
        }

        for(int mask = 16; mask >= 1; mask >>=1)
        {
            divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        }

        if(lane_id == 0)
        {
            reduction[warp_id] = divisor;
        }
        __syncthreads();

        if(warp_id == 0)
        {
            divisor = ty < BLOCK_DIM_Y / 32 ? reduction[ty] : 0;
            for(int mask = 16; mask >= 1; mask >>=1)
            {
                divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
            }
        }
        if(ty == 0)
        {
            reduction[0] = divisor;
        }

        for(int i = ty; i < w / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(a[row * w + i * 4])[0];
            val.x = __expf(val.x - maxval) / divisor;
            val.y = __expf(val.y - maxval) / divisor;
            val.z = __expf(val.z - maxval) / divisor;
            val.w = __expf(val.w - maxval) / divisor;

            reinterpret_cast<float4*>(b[row * w + i * 4])[0] = val;
            
        }


   }

}

template<typename scalar_t>
__global__ void softmax_kernel_by_coalsced_warp_combine(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
    int row = blockIdx.x;
    int ty = threadIdx.y;
    int warp_id = ty / 32;
    int lane_id = ty % 32;
    
    float reduction_maxval[BLOCK_DIM_Y / 32];
    float reduction_div[BLOCK_DIM_Y / 32];

    if(row < h)
    {
        float maxval =  -INFINITY;
        float premaxval =  -INFINITY;
        float divisor = 0.0f;

        for(int i = ty; i < w / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(&a[row * w + 4*i])[0];
            maxval = fmaxf(maxval, val.x);
            maxval = fmaxf(maxval, val.y);
            maxval = fmaxf(maxval, val.z);
            maxval = fmaxf(maxval, val.w);
            if(maxval > premaxval)
            {
                divisor *= __expf(premaxval - maxval);
                premaxval = maxval; 
            }
            divisor += __expf(val.x - maxval);
            divisor += __expf(val.y - maxval);
            divisor += __expf(val.z - maxval);
            divisor += __expf(val.w - maxval);
        }
        float dis_max = 0.0f;
        float dis_divisor = 0.0f;
        #pragma unroll URF
        for(int mask = 16; mask >= 1; mask >>= 1)
        {
            dis_max = __shfl_xor_sync(0xffffffff,maxval,mask,32);
            dis_divisor = __shfl_xor_sync(0xffffffff,divisor,mask,32);
            if(maxval < dis_max)
            {
                divisor *= __expf(maxval - dis_max);
                maxval = dis_max;
            }
            else
            {
                dis_divisor *= __expf(dis_max - maxval);
            }
            divisor += dis_divisor;
        }

        if(ty == 0)
        {
            reduction_maxval[warp_id] = maxval;
            reduction_div[warp_id] = divisor;
        }
        __syncthreads();


            #pragma unroll
            for(int mask = 16; mask >= 1; mask >>=1)
            {
                dis_max = __shfl_xor_sync(0xffffffff,maxval,mask,32);
                dis_divisor = __shfl_xor_sync(0xffffffff,divisor,mask,32);
                if(maxval < dis_max)
                {
                    divisor *= __expf(maxval - dis_max);
                    maxval = dis_max;
                }
                else
                {
                    dis_divisor *= __expf(dis_max - maxval);
                }
                divisor += dis_divisor;
            }
       
        if(lane_id == 0)
        {
            reduction_maxval[0] = maxval;
            reduction_div[0] = divisor;
        }

        __syncthreads();
        maxval = reduction_maxval[0];
        divisor = reduction_div[0];
        //printf("线程id: %d 最大值: %f  除数：%f\n",ty,maxval,divisor);

        for(int i = ty; i < w / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4*>(&a[row * w + i * 4])[0];
            //printf("合并化 ty:%d x: %f y: %f \n",i,val.x,val.y);
            val.x = __expf(val.x - maxval) / divisor;
            val.y = __expf(val.y - maxval) / divisor;
            //printf("softmax求值 ty:%d x: %f y: %f \n",i,val.x,val.y);
            val.z = __expf(val.z - maxval) / divisor;
            val.w = __expf(val.w - maxval) / divisor;
            reinterpret_cast<float4*>(&b[row * w + i* 4])[0] = val;
        }

    }
}




int main()
{
     constexpr int m = 32;
     constexpr int n = 4;

    //float c[m];
    //float d[n];
    float* c = new float[m*n];
    float* d = new float[n*m];

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0,1.0);

    for(int i = 0; i < m*n; i++)
    {
        c[i] = i;
    }
    for(int i = 0; i < 32; i++)
    {
        //std::cout<<"原始值" <<c[i]<<std::endl;
    }

    float* d_c;
    float* d_d;
    cudaMalloc(&d_c, m*n*sizeof(float));
    cudaMalloc(&d_d, m*n*sizeof(float));

    cudaMemcpy(d_c,c,m*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dim3 blockdim(1,8,1);
    dim3 griddim(n,1,1);// softmax_kernel_by_coalsced_warp_combine griddim(h, 1, 1)确保每个block处理一行

    softmax_kernel_by_coalsced_warp_combine<float><<<griddim,blockdim>>>(d_c,d_d,m,n);
    //softmax_kernel<float><<<griddim,blockdim>>>(d_c,d_d,m,n);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cerr<<cudaGetErrorString(err)<<std::endl;
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(d,d_d,m*n*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    float sum[n] ={0.0f};
    for(int j = 0; j < n; j++ )
    for(int i = j * m; i < m * (n -3 + j); i++)
    {
        sum[j] += d[i];
    }
    std::cout<<"求和： "<< std::endl
    <<"第1行: "<<sum[0]<<std::endl
    <<"第2行: "<<sum[1]<<std::endl
    <<"第3行: "<<sum[2]<<std::endl
    <<"第4行: "<<sum[3]<<std::endl;
     
    return 0;
}
