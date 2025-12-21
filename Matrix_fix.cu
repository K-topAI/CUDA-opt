#include<cuda_runtime.h>
#include<cassert>
#include<vector>
#include<iostream>
#include<iomanip>
#include<functional>
#include<stdio.h>
#include<algorithm>
#include<cuda_bf16.h>
#include<cuda_fp16.h>
#include<mma.h>
#define CHECK_CUDA_ERROR(val) check((val),#val, __FILE__, __LINE__)
template<class T>
void check(T err, const char* const func, const char* const file, int const line)
{
    if(err != cudaSuccess)
    {
        std::cerr << "Cuda Runtime erro at:  " << file << " " << line 
                    <<std::endl;
        std::cerr <<cudaGetErrorString(err)<< " " << func << std::endl;
        std::exit(0); 
    }
}
#define CHECK_LAST_ERROR() check_last_error(__FILE__,__LINE__)
void check_last_error(const char* const file, int const line )
{
    cudaError_t const err{cudaGetLastError()};
    if(err != cudaSuccess)
    {
        std::cerr << "Cuda runtime Last Error at:  " << file << " "
                        << line<<std::endl; 
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(0);
    }

}
template <typename T>
__global__ void gemm_v0(size_t m, size_t n, size_t k, T alpha, T const* A
                        , size_t lda, T const* B, size_t ldb, T beat, T* C, size_t ldc)
{
    size_t const C_row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const C_col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if(C_row_idx < m && C_col_idx < n)
    {
        T sum = static_cast<T>(0);

        for(size_t k_idx = 0; k_idx < k; ++k_idx)
        {//每个线程 计算出矩阵的一个结果
            sum += A[C_row_idx * lda + k_idx] * B[C_col_idx * ldb + k_idx];
        } 
        //按行存回矩阵
        C[C_row_idx * ldc + C_col_idx] = alpha * sum + beat * C[C_row_idx * ldc + C_col_idx];

    }

}

template <typename T>
void launch_gemm_kernel_v0(size_t m, size_t n, size_t k, size_t alpha, 
                            T const* A, size_t lda, T const* B, size_t ldb,
                            size_t beat, T* C, size_t ldc, cudaStream_t stream)
{
    dim3 const block_dim(32U,32U,1U);
    dim3 const grid_dim((static_cast<unsigned int>(m) + block_dim.x -1U) / block_dim.x ,
                         (static_cast<unsigned int>(n) + block_dim.y -1u) /block_dim.y, 1U);
    
    gemm_v0<T><<<grid_dim,block_dim,0U,stream>>>(m, n, k, alpha, A, lda, B, ldb, beat, C, ldc, stream);
    CHECK_LAST_ERROR();
}
template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T  beat, T* C, size_t ldc)
{
    size_t const C_row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    size_t const C_col_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(C_row_idx < m && C_col_idx <n)
    {
        T sum = static_cast<T>(0);

        for(size_t k_idx = 0; k_idx < k; k_idx++)
        {
            sum += A[k_idx * lda + C_row_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] = alpha * sum  + beat * C[C_row_idx * ldc + C_col_idx];
    }
}
template<typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k, size_t alpha, 
                            T const *A, size_t lda, T const* B, size_t ldb,
                            size_t beat, T* C, size_t ldc, cudaStream_t stream)
{
    dim3 block_dim {32U, 32U, 1U};
    dim3 grid_dim  {(static_cast<T>(m) + block_dim.x -1U) / block_dim.x,
                     (static_cast<T>(n) + block_dim.y -1u) / block_dim.y,
                     1u};
    
    gemm_v01<T><<<grid_dim, block_dim>>>(m, n, k, alpha, A, lda, 
                                        B, ldb, beat, C, ldc);
    CHECK_LAST_ERROR();
}

template<typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, 
        size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_X][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n, size_t k)
{

#pragma unroll
    for(size_t load_idx = 0u; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + 
    NUM_THREADS - 1U) / NUM_THREADS; ++load_idx)
    {
        // A_tile : BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K
        // 找到tile中的坐标值
        // 行号： 跨行/ ; 列号：在同一行内的某一列求%
        size_t const A_thread_tile_row_idx = (thread_linear_idx + NUM_THREADS * load_idx)
                                             / BLOCK_TILE_SIZE_K;
        size_t const A_thread_tile_col_idx = (thread_linear_idx + NUM_THREADS * load_idx)
                                             % BLOCK_TILE_SIZE_K;

        //找到block中的坐标值
        size_t const A_row_idx = blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_tile_row_idx;
        size_t const A_col_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_K 
                                + A_thread_tile_col_idx;

        T val = static_cast<T>(0);
        if(A_row_idx < m && A_col_idx < k)
        {
            //找到转递到共享内存中的值
            val = A[A_row_idx * lda + A_col_idx];
        }

    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);

    A_thread_block_tile[A_thread_tile_row_idx][A_thread_tile_col_idx] = val;

    }

#pragma unroll
    for(size_t load_idx = 0u;
        load_idx < (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K + 
        NUM_THREADS -1U) / NUM_THREADS; ++load_idx)
        {
            //B_tile尺寸 BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K
            size_t const B_thread_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) 
                                                / BLOCK_TILE_SIZE_X;
            size_t const B_thread_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) 
                                                % BLOCK_TILE_SIZE_X; 

            size_t const B_row_idx = thread_block_tile_idx * BLOCK_TILE_SIZE_K 
                                    + B_thread_tile_row_idx;
            size_t const B_col_idx = BLOCK_TILE_SIZE_X * blockIdx.x 
                                    + B_thread_tile_col_idx;

            T val = static_cast<T>(0);

            if(B_row_idx < k && B_col_idx < n)
            {
                val = B[B_row_idx * ldb + B_col_idx];//注意和A进行比较
            }
        
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K %NUM_THREADS == 0U);

        B_thread_block_tile[B_thread_tile_row_idx][B_thread_tile_col_idx] = val;

        }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T beat,T* C, size_t ldc)
{

    //计算时直接从共享内存中取出数据，计算后写回主存？
    constexpr size_t NUM_THREADS = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y;
    size_t const thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    size_t const C_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const C_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles = (k + BLOCK_TILE_SIZE_K -1) / BLOCK_TILE_SIZE_K;//向上取整数 圆整方式是rp 注意观察SASS 和PTX中的圆整方式
    T sum = static_cast<T>(0);
     for(size_t thread_block_tile_idx = 0u;
            thread_block_tile_idx < num_thread_block_tiles;
            ++thread_block_tile_idx)
            {
                load_data_to_shared_memory<T,BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                 BLOCK_TILE_SIZE_K, NUM_THREADS>(A, lda, B, ldb,
                                                A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx,
                                                 thread_linear_idx, m, n, k);
                 __syncthreads();

               #pragma unroll
                for(size_t k_i = 0u; k_i < BLOCK_TILE_SIZE_K; ++k_i)
                    {
                        sum += A_thread_block_tile[threadIdx.y][k_i] * B_thread_block_tile[k_i][threadIdx.x];
                    }
                __syncthreads();
            }

    if (C_row_idx < m && C_col_idx < n)
        {
         C[C_row_idx * ldc + C_col_idx] = alpha * sum + beat * C[C_row_idx * ldc + C_col_idx];
        }

}

template<typename T>
void launch_gemm_kernel_v02(size_t m,size_t n, size_t k, T alpha,
                            T const* A, size_t lda, T const* B, size_t ldb, 
                            T beat, T* C, size_t ldc, cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_X = 32U;
    constexpr unsigned int BLOCK_TILE_Y = 32u;
    constexpr unsigned int BLOCK_TILE_K = 32U;
    constexpr unsigned int NUM_THREADS = BLOCK_TILE_X * BLOCK_TILE_Y;
    static_assert(BLOCK_TILE_K * BLOCK_TILE_Y % NUM_THREADS == 0u);
    static_assert(BLOCK_TILE_K * BLOCK_TILE_X % NUM_THREADS == 0U);

    dim3 const block_dim = {BLOCK_TILE_X, BLOCK_TILE_Y, 1u};
    dim3 const grid_dim ={(static_cast<unsigned int >(n) + block_dim.x -1u) / block_dim.x,
                          (static_cast<unsigned int>(m) + block_dim.y -1U) / block_dim.y,
                           1U};
    gemm_v02<T,BLOCK_TILE_X,BLOCK_TILE_Y,BLOCK_TILE_K><<<grid_dim,block_dim,0,stream>>>(m,n,k,alpha,A,lda,B,ldb,beat,C,ldc);

}


//将数据从shared memory传入到 register 中进行计算的 其中每次传入数据的的长度为 THREAD_TILE_SIZE_Y
template<typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y, size_t BLOCK_TILE_K,
         size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v03(size_t m, size_t n, size_t k, T alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T beat, T  *C, size_t ldc)
{
    //参与的线程总数 二维网格的大小：BLOCK_TILE_X * BLOCK_TILE_Y 
    constexpr size_t NUM_THREADS = BLOCK_TILE_X * BLOCK_TILE_Y / THREAD_TILE_SIZE_Y;

    //全局的线性id
    size_t const thread_linear_idx = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ T A_thread_block_tile[BLOCK_TILE_Y][BLOCK_TILE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_K][BLOCK_TILE_X];

    size_t const num_thread_block_tiles =(k + BLOCK_TILE_K -1) / BLOCK_TILE_K;


    T C_thread_results[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

    for(size_t thread_block_tile_idx =0u;
        thread_block_tile_idx < num_thread_block_tiles;
        ++thread_block_tile_idx)
        {
            load_data_to_shared_memory<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K,NUM_THREADS>(
                A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile, thread_block_tile_idx, 
                m , n, k );
            __syncthreads();//在thread 完成操作后
#pragma unroll 
            for(size_t k_i =0u; k_i < BLOCK_TILE_K; ++k_i)
            {
                size_t B_Thread_block_tile_row_idx = k_i;
                // 问
                T const B_val = B_thread_block_tile[B_Thread_block_tile_row_idx][thread_linear_idx % BLOCK_TILE_X];

                #pragma unroll
                for(size_t thread_tile_row_idx = 0u;
                    thread_tile_row_idx < THREAD_TILE_SIZE_Y; 
                    ++thread_tile_row_idx)
                    {//传入到寄存器的只有一列（行）且该长度为 THREAD_TILE_SIZE_Y
                        size_t const A_thread_block_tile_row_idx = (thread_linear_idx / BLOCK_TILE_X) * THREAD_TILE_SIZE_Y
                                                                   + thread_tile_row_idx;
                        size_t const A_thread_block_tile_col_idx = k_i;

                        T const  A_val = A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx];
                        C_thread_results[thread_tile_row_idx] += A_val * B_val;
                      
                    }

            }
            __syncthreads();
        }

#pragma unroll
    for(size_t thread_tile_row_idx = 0U;
        thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
        {
            //每个线程就是持有那么多数据？
            size_t const C_row_idx = blockIdx.y * BLOCK_TILE_Y + thread_linear_idx / BLOCK_TILE_X * THREAD_TILE_SIZE_Y 
                                    + thread_tile_row_idx;
            size_t const C_col_idx = blockIdx.x * BLOCK_TILE_X + thread_linear_idx % BLOCK_TILE_X;

            if(C_row_idx < m && C_col_idx < n)
            {
                C[C_row_idx * ldc + C_col_idx] = alpha * C_thread_results[thread_tile_row_idx] + 
                                                beat * C[C_row_idx * ldc + C_col_idx];
            }

        }

}

template<typename T>
void launch_geem_kernel_v03(size_t m, size_t n, size_t k, T const* alpha, T const* A, size_t lda,
                             T const* B, size_t ldb, T const* beta, T* C, size_t ldc, cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_X = 64;
    constexpr unsigned int BLOCK_TILE_Y = 64;
    constexpr unsigned int BLOCK_TILE_K = 8;

    constexpr unsigned int THREAD_TILE_SIZE_Y = 8;
    //数据处理时求X行X列的依据
    constexpr unsigned int NUM_THREADS_PER_BLOCK = BLOCK_TILE_Y * BLOCK_TILE_X / THREAD_TILE_SIZE_Y;

    static_assert(BLOCK_TILE_Y % THREAD_TILE_SIZE_Y == 0);

    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_X == 0);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_K == 0);

    dim3 const block_dim = { NUM_THREADS_PER_BLOCK, 1, 1};
    dim3 const grid_dim = {(static_cast<unsigned int>(m) + BLOCK_TILE_X -1) / BLOCK_TILE_X,
                            (static_cast<unsigned int>(n) + BLOCK_TILE_Y -1) / BLOCK_TILE_Y,
                            1};
    gemm_v03<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K, THREAD_TILE_SIZE_Y><<<grid_dim,block_dim, 0, stream>>>(m, n, k, *alpha
    , A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_ERROR();

}

template<typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y,
                     size_t BLOCK_TILE_K, size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v04(size_t m, size_t n,size_t k, T alpha, T const* A, size_t lda,
                        T const* B, size_t ldb,  T beta,T* C, size_t ldc)
{
    constexpr size_t BLOCK_TILE_SUM = BLOCK_TILE_X * BLOCK_TILE_Y;
    constexpr size_t THREAD_TILE_SUM = THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y;
    constexpr size_t NUM_THREADS = BLOCK_TILE_SUM / THREAD_TILE_SUM;

    size_t const thread_linear_idx = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ T A_thread_block_tile[BLOCK_TILE_Y][BLOCK_TILE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_K][BLOCK_TILE_X];


    size_t const num_threads_block_tiles = (k + BLOCK_TILE_K -1) / BLOCK_TILE_K;

    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = { static_cast<T>(0) };
    
    T A_val[THREAD_TILE_SIZE_Y] = { 0U };
    T B_val[THREAD_TILE_SIZE_X] = { 0U };

    for(size_t thread_block_tile_idx = 0u;
        thread_block_tile_idx < num_threads_block_tiles; 
        ++thread_block_tile_idx)
        {
            load_data_to_shared_memory<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K, NUM_THREADS>(A, lda, B, ldb,
             A_thread_block_tile, B_thread_block_tile, m, n, k);
             __syncthreads();

             #pragma unroll
             for(size_t k_i = 0u; k < BLOCK_TILE_K; ++k_i)
             {

                //矩阵A_tile from shared memory取值 
                size_t const A_thread_block_tile_row_idx = thread_linear_idx / (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y;
                size_t A_thread_block_col_idx = k_i;

                #pragma unroll

                for(size_t thread_tile_row_idx = 0u; thread_tile_row_idx < THREAD_TILE_SIZE_Y;++thread_tile_row_idx)
                {
                    A_val[thread_tile_row_idx] = A_thread_block_tile[A_thread_block_tile_row_idx + thread_tile_row_idx][A_thread_block_col_idx];
                }

                //矩阵B_tile取值

                size_t  B_thread_block_tile_row_idx = k_i;
                size_t const B_thread_block_tile_col_idx = thread_linear_idx % (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X;

                #pragma unroll
                for(size_t thread_tile_col_idx = 0u;
                    thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
                    {
                        B_val[thread_tile_col_idx] = B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx];

                    }
                
                for(size_t thread_tile_row_idx = 0u; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
                {
                    for(size_t thread_tile_col_idx = 0u; thread_tile_col_idx < THREAD_TILE_SIZE_X;++thread_tile_col_idx)
                    {
                        C_thread_results[thread_tile_row_idx][thread_tile_col_idx] += A_val[thread_tile_row_idx] *B_val[thread_tile_col_idx];
                    }
                }
             }
             __syncthreads();
    
        }

        //from register -> global
        for(size_t thread_tile_row_idx = 0u; thread_tile_row_idx < THREAD_TILE_SIZE_Y ; ++thread_tile_row_idx)
        {
            for(size_t thread_tile_col_idx = 0u; thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
            {
                size_t C_row_idx = blockIdx.y * BLOCK_TILE_Y + threadIdx.x % (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y + thread_tile_row_idx;
                size_t C_col_idx = blockIdx.x * BLOCK_TILE_Y + threadIdx.y % (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X + thread_tile_col_idx;

                if(C_row_idx < m && C_col_idx < n)
                {
                    C[C_row_idx * ldc + C_col_idx] = alpha* C_thread_results[thread_tile_row_idx][thread_tile_col_idx] + beta * C[C_row_idx * ldc + C_col_idx];
                }
            }
        }

}
template<typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k, T alpha,
                            T const * A, size_t lda, T const* B, size_t ldb,T beat,
                            T C, size_t ldc, cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_X = 128;
    constexpr unsigned int BLOCK_TILE_Y = 128;
    constexpr unsigned int BLOCK_TILE_K = 16;

    constexpr unsigned int THREAD_TILE_SIZE_X = 8;
    constexpr unsigned int THREAD_TILE_SIZE_Y = 8;

    //每个线程处理的数据大小是THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y
    constexpr unsigned int NUM_THREADS_PER_BLOCK = (BLOCK_TILE_X * BLOCK_TILE_Y) / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y);
     static_assert(BLOCK_TILE_X % THREAD_TILE_SIZE_X == 0);
     static_assert(BLOCK_TILE_Y % THREAD_TILE_SIZE_Y == 0);
     static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_K == 0);
     static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_X == 0);

     static_assert(BLOCK_TILE_X * BLOCK_TILE_K % NUM_THREADS_PER_BLOCK == 0);
     static_assert(BLOCK_TILE_Y * BLOCK_TILE_K % NUM_THREADS_PER_BLOCK == 0);

     dim3 const block_dim = { NUM_THREADS_PER_BLOCK, 1 ,1};
     dim3 const grid_dim = {(static_cast<unsigned int>(m) + BLOCK_TILE_X - 1) / BLOCK_TILE_X,
                            (static_cast<unsigned int>(n) + BLOCK_TILE_Y -1) / BLOCK_TILE_Y,
                            1};
    gemm_v04<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K, 
    THREAD_TILE_SIZE_X,THREAD_TILE_SIZE_Y><<<grid_dim, block_dim, 0, stream>>>
                                    (m ,n, k, alpha, A, lda, B, ldb, beat, C, ldc);
    CHECK_LAST_ERROR();
}


template<typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y, size_t BLOCK_TILE_K,
            size_t NUM_THREADS,size_t BLOCK_TILE_SKEW_X = 0, size_t BLOCK_TILE_SKEW_Y = 0,
             typename VECTOR_TYPE = int4>
__device__ void load_data_to_shared_memory_transposed_vectorized(T const* A, size_t lda,
                                                                 T const* B, size_t ldb,
                                                                 T A_thread_block_tile_transposed[BLOCK_TILE_K][BLOCK_TILE_Y + BLOCK_TILE_SKEW_Y],
                                                                 T B_thread_block_tile[BLOCK_TILE_K][BLOCK_TILE_X + BLOCK_TILE_SKEW_X],
                                                                 size_t thread_block_tile_idx,
                                                                 size_t thread_linear_idx,
                                                                 size_t m ,size_t n, size_t k 
                                                                 )
{ 
    constexpr size_t NUM_VECTOR_UNITS = sizeof(VECTOR_TYPE) / sizeof(int);
    static_assert(sizeof(VECTOR_TYPE) % sizeof(int) == 0);

    static_assert(BLOCK_TILE_K % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_X % NUM_VECTOR_UNITS == 0);

    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K = BLOCK_TILE_K / NUM_VECTOR_UNITS;
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X = BLOCK_TILE_X / NUM_VECTOR_UNITS;
    
    static_assert(BLOCK_TILE_X * sizeof(T) % sizeof(VECTOR_TYPE) == 0);
    static_assert(BLOCK_TILE_K * sizeof(T) % sizeof(VECTOR_TYPE) == 0);

   //保证对齐是正确的
    static_assert((BLOCK_TILE_X + BLOCK_TILE_SKEW_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_Y + BLOCK_TILE_SKEW_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

#pragma unroll
    for(size_t load_idx =0; load_idx < (NUM_THREADS + BLOCK_TILE_Y * VECTORIZED_BLOCK_TILE_SIZE_K -1) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_BLOCK_TILE_SIZE_K;
        
        size_t const A_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS;

        size_t const A_row_idx = blockIdx.y * BLOCK_TILE_Y + A_thread_block_tile_row_idx;

        size_t const A_col_idx = thread_block_tile_idx  * BLOCK_TILE_K + A_thread_block_tile_col_idx;

        int4 A_row_vector_vals{0,0,0,0};
         if( A_row_idx < m && A_col_idx < k)
         {
            A_row_vector_vals = *reinterpret_cast<int4 const*>(&A[A_row_idx * lda + A_col_idx]);
         }
         //处理超出边界的值
         if(A_col_idx + NUM_VECTOR_UNITS > k)
         {
            size_t const num_invalid_element = A_col_idx + NUM_VECTOR_UNITS - k;

            T* const A_row_vector_vals_ptr = reinterpret_cast<T*>(&A_row_vector_vals);
            for(size_t k_idx = 0; k_idx < num_invalid_element; ++k_idx)
            {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1 - k_idx] = static_cast<T>(0); 
            }
         }

         if(A_thread_block_tile_row_idx < BLOCK_TILE_Y && A_thread_block_tile_col_idx < BLOCK_TILE_K)
         {
            //转置后需要存储到不同的行
             for(size_t i =0 ;i < NUM_VECTOR_UNITS; i++ )
                 A_thread_block_tile_transposed[A_thread_block_tile_row_idx + i][A_thread_block_tile_col_idx] = reinterpret_cast<T const*>(&A_row_vector_vals)[i];
         }


    }
#pragma unroll
    for(size_t load_idx = 0; load_idx < (NUM_THREADS + BLOCK_TILE_K * VECTORIZED_BLOCK_TILE_SIZE_X -1) / BLOCK_TILE_K; ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx = (thread_linear_idx + load_idx * NUM_THREADS) / VECTORIZED_BLOCK_TILE_SIZE_X;
        size_t const B_thread_block_tile_col_idx = (thread_linear_idx + load_idx * NUM_THREADS) % VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS;

        size_t const B_row_idx = thread_block_tile_idx * BLOCK_TILE_K + B_thread_block_tile_row_idx;
        size_t const B_col_idx = blockIdx.x * BLOCK_TILE_X + B_thread_block_tile_col_idx;

        int4 B_vector_vals = {0, 0, 0 ,0};
        if(B_row_idx < k && B_col_idx < n)
        {
            B_vector_vals = *reinterpret_cast<int4 const*>(&B[B_row_idx * ldb + B_col_idx]);

            if(B_col_idx + NUM_VECTOR_UNITS > n )
            {
                size_t const num_invalid_element = B_col_idx + NUM_VECTOR_UNITS - n;
                T* const B_row_vector_vals_ptr = reinterpret_cast<T*>(&B_vector_vals);
                 for(size_t i = 0; i < num_invalid_element; ++i)
                 {
                        B_row_vector_vals_ptr[NUM_VECTOR_UNITS -1 -i] = static_cast<T>(0);
                 }
                    
            }
        }

        if(B_thread_block_tile_row_idx < BLOCK_TILE_K && B_thread_block_tile_col_idx < BLOCK_TILE_X)
        {
            *reinterpret_cast<int4*>(&B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx])
             = B_vector_vals;
        }
    }

}
//在寄存器中执行A_tile * B_tile(一维)
template<typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y ,size_t BLOCK_TILE_K,
                 size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y, typename VectorType>
__global__ void gemm_v05(size_t m , size_t n, size_t k, T alpha, T const* A, size_t lda,
                        T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
    constexpr size_t NUM_THREADS = BLOCK_TILE_X * BLOCK_TILE_Y /(THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y);
    size_t const thread_linear_idx = threadIdx.y* blockDim.x + threadIdx.x;

    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_K][BLOCK_TILE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_K][BLOCK_TILE_X];
    size_t const num_thread_block_tiles = (k + BLOCK_TILE_K -1) / BLOCK_TILE_K;

    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    constexpr size_t NUM_VECTOR_UNITS = sizeof(int4) / sizeof(T);

    static_assert(sizeof(int4) % sizeof(T) == 0u);
    static_assert(BLOCK_TILE_K % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_X % NUM_VECTOR_UNITS == 0);
    
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X = THREAD_TILE_SIZE_X / NUM_VECTOR_UNITS;
    
    static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    for(size_t thread_block_tile_idx = 0u;
        thread_block_tile_idx < num_thread_block_tiles; ++thread_block_tile_idx)
        {
            //逐行加载到shared memory
            load_data_to_shared_memory_transposed_vectorized<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K,
            NUM_THREADS,THREAD_TILE_SIZE_X,THREAD_TILE_SIZE_Y>(A,lda,B,ldb,A_thread_block_tile_transposed,B_thread_block_tile,thread_block_tile_idx,thread_linear_idx,m,n,k);
            __syncthreads();
#pragma unroll
            for(size_t k_i; k_i < BLOCK_TILE_K; ++k_i)
            {//#affine_map (i,k,j) ->(i,j) 线程块的组织方式是由输出的矩阵的维度决定的
                size_t const A_thread_block_tile_row_idx = thread_linear_idx / (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y;
                size_t const A_thread_block_tile_col_idx = k_i;
#pragma unroll
                for(size_t thread_tile_row_idx = 0; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
                {
                    //A_threa_block_tile can't access by vectorized
                    A_vals[thread_tile_row_idx] = A_thread_block_tile_transposed[A_thread_block_tile_col_idx][A_thread_block_tile_row_idx + thread_tile_row_idx];
                

                }

                size_t const B_thread_block_tile_row_idx = k_i;
                size_t const B_thread_block_tile_col_idx = thread_linear_idx % (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X;
#pragma unroll 
                for(size_t thread_tile_col_vector_idx = 0; thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X; ++thread_tile_col_vector_idx)
                {

                    *reinterpret_cast<int4*>( &B_vals[thread_tile_col_vector_idx * NUM_VECTOR_UNITS]) = *reinterpret_cast<int4*>(B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx + thread_tile_col_vector_idx * NUM_VECTOR_UNITS]);
                }

                for(size_t thread_tile_row_idx = 0; thread_tile_row_idx < THREAD_TILE_SIZE_X; ++thread_tile_row_idx)
                {
                    for(size_t thread_tile_col_idx = 0; thread_tile_col_idx < THREAD_TILE_SIZE_Y; ++thread_tile_col_idx)
                    {
                        C_thread_results[thread_tile_row_idx][thread_tile_col_idx] = A_vals[thread_tile_row_idx] * B_vals[thread_tile_col_idx];

                    }
                }

            }


        __syncthreads();
        }

        for(size_t thread_tile_row_idx = 0; thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
        {
            for(size_t thread_tile_col_vector_idx = 0; thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X; ++thread_tile_col_vector_idx)
            {
                //affine_map (shared) -> DRAM
                size_t const C_row_idx = blockIdx.y * BLOCK_TILE_Y + thread_linear_idx / (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y + thread_tile_row_idx;
                 size_t const C_col_idx = blockIdx.x * BLOCK_TILE_X + thread_linear_idx % (BLOCK_TILE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X + thread_tile_col_vector_idx * NUM_VECTOR_UNITS;
            
                int4 C_row_vector_vals = *reinterpret_cast<int4*>(&C[C_row_idx * ldc + C_col_idx]);

                int4 const C_thread_results_row_vector_vals = *reinterpret_cast<int4 const*>(&C_thread_results[thread_tile_row_idx][thread_tile_col_vector_idx]);

            for(size_t i= 0; i < NUM_VECTOR_UNITS; i++)
            {
                reinterpret_cast<T*>(&C_row_vector_vals)[i] = alpha * reinterpret_cast<T const*>(&C_thread_results_row_vector_vals)[i] + beta * reinterpret_cast<T const*>(&C_row_vector_vals)[i];
            }

            if(C_row_idx < m && C_col_idx < n)
            {
                *reinterpret_cast<int4*>(&C[C_row_idx * ldc + C_col_idx]) = C_row_vector_vals; 
            }
            
            }


        }
}
template<typename T>
void launch_gemm_kernel_V05(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda,
                            T const* B, size_t ldb, T beta, T* const C, size_t ldc, cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_X = 128;
    constexpr unsigned int BLOCK_TILE_Y = 128;
    constexpr unsigned int BLOCK_TILE_K = 16;

    constexpr unsigned int THREAD_TILE_SIZE_X = 8;
    constexpr unsigned int THREAD_TILE_SIZE_Y = 8;

    constexpr unsigned int NUM_THREADS_PER_BLOCK = BLOCK_TILE_X * BLOCK_TILE_Y / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y);
    
    static_assert(BLOCK_TILE_X % THREAD_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_Y % THREAD_TILE_SIZE_Y == 0);

    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_K == 0);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_X == 0);

    static_assert(BLOCK_TILE_X * BLOCK_TILE_K % NUM_THREADS_PER_BLOCK == 0);
    static_assert(BLOCK_TILE_Y * BLOCK_TILE_K % NUM_THREADS_PER_BLOCK == 0);

    dim3 block_dim = {NUM_THREADS_PER_BLOCK , 1, 1};
    dim3 grid_dim = {(static_cast<unsigned int>(n) + BLOCK_TILE_X -1) / BLOCK_TILE_X,
                    (static_cast<unsigned int>(m) + BLOCK_TILE_Y -1) / BLOCK_TILE_Y,
                    1U};
    
    gemm_v05<T,BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K,THREAD_TILE_SIZE_X,THREAD_TILE_SIZE_Y><<<grid_dim,block_dim,0,stream>>>(m ,n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    CHECK_LAST_ERROR();
}


#define WARP_SIZE 32

template<typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y, size_t BLOCK_TILE_K, size_t WARP_TILE_SIZE_X,
        size_t WARP_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y, size_t NUM_THREADS_PER_WARP_X,
        size_t NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v06(size_t m , size_t n, size_t k, T alpha,
                        T const* A, size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc)
{
     constexpr unsigned int NUM_WARPS_X = BLOCK_TILE_X / WARP_TILE_SIZE_X;
     constexpr unsigned int NUM_WARPS_Y = BLOCK_TILE_Y / WARP_TILE_SIZE_Y; 


     constexpr unsigned int NUM_TILES_PER_WARP_X = WARP_TILE_SIZE_X / THREAD_TILE_SIZE_X;
     constexpr unsigned int NUM_TILES_PER_WARP_Y = WARP_TILE_SIZE_Y / THREAD_TILE_SIZE_Y; 

   
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X = NUM_TILES_PER_WARP_X / NUM_THREADS_PER_WARP_X;
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y = NUM_TILES_PER_WARP_Y / NUM_THREADS_PER_WARP_Y;
     //constexpr unsigned int NUM_THRERAD_TILES_PER_WARP_X = WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X);

    constexpr unsigned int NUM_THREADS_X = NUM_WARPS_X * NUM_THREAD_TILES_PER_WARP_X;
    constexpr unsigned int NUM_THREADS_Y = NUM_WARPS_Y * NUM_THREAD_TILES_PER_WARP_Y;

    constexpr size_t  NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y;

    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_K][BLOCK_TILE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_K][BLOCK_TILE_X];

    //在register中的缓存区
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    //每个线程进入之后找到其在该块内的线性Idx
    size_t const thread_linear_idx = blockDim.x * threadIdx.y + threadIdx.x;//当前线程对应的线性ID
    size_t const warp_linear_idx = thread_linear_idx / WARP_SIZE;//当前线程所在的第n个warp的线性Id
    
    size_t const warp_row_idx =warp_linear_idx / NUM_THREADS_X;
    size_t const warp_col_idx =warp_linear_idx % NUM_THREADS_X;

    size_t const thread_linear_idx_in_warp = thread_linear_idx  % WARP_SIZE;//第n个warp 中的线程的线性id
    size_t const thread_linear_row_idx_in_warp = thread_linear_idx_in_warp / NUM_THREADS_PER_WARP_X;
    size_t const thread_linear_col_idx_in_warp = thread_linear_idx_in_warp % NUM_THREADS_PER_WARP_X;


    size_t const num_thread_block_tiles = (k + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    constexpr size_t NUM_VECTOR_UNITS = sizeof(int4) / sizeof(T);

    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X = THREAD_TILE_SIZE_X / NUM_VECTOR_UNITS;
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_Y = THREAD_TILE_SIZE_Y / NUM_VECTOR_UNITS;

    for(size_t thread_block_tiles_idx = 0; thread_block_tiles_idx < num_thread_block_tiles; ++thread_block_tiles_idx)
    {
        load_data_to_shared_memory_transposed_vectorized<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K, NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
        B_thread_block_tile, thread_block_tiles_idx, thread_linear_idx,m,n,k);
        __syncthreads();
    }

     for(size_t k_i = 0; k_i < BLOCK_TILE_K; ++k_i)
     {
        for(size_t thread_tile_warp_row_idx = 0; thread_tile_warp_row_idx < NUM_THREAD_TILES_PER_WARP_Y; ++thread_tile_warp_row_idx)
        {
            //  在warp的中块的id + 其中一个线程的id * warp中某一块的行 + 当前线程行 * 行数
           size_t  const A_thread_block_tile_transposed_row_idx = warp_row_idx * WARP_TILE_SIZE_Y + 
                                                                    thread_tile_warp_row_idx *  (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                                                                    thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y;
            size_t const A_thread_block_tile_transposed_col_idx = k_i;

            for(size_t thread_tile_y_vector_idx =0; thread_tile_y_vector_idx < VECTORIZED_THREAD_TILE_SIZE_Y; ++thread_tile_y_vector_idx)
            {
                *reinterpret_cast<int4*>(&A_vals[thread_tile_warp_row_idx][thread_tile_y_vector_idx * NUM_VECTOR_UNITS])
                 = *reinterpret_cast<int4*>(&A_thread_block_tile_transposed[A_thread_block_tile_transposed_col_idx][A_thread_block_tile_transposed_row_idx + thread_tile_y_vector_idx * NUM_VECTOR_UNITS]);
            }
        }
        //Load B
        for(size_t thread_tile_warp_col_idx = 0; thread_tile_warp_col_idx < NUM_THREAD_TILES_PER_WARP_X; ++thread_tile_warp_col_idx)
        {
            size_t const B_thread_tile_row_idx = k_i;
            size_t const B_thread_tile_col_idx = warp_col_idx * WARP_TILE_SIZE_X +
                                                thread_tile_warp_col_idx * (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                                                thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X;
            for(size_t thread_tile_x_vector_idx = 0; thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X; ++thread_tile_x_vector_idx)
            {
                *reinterpret_cast<int4*>(&B_vals[thread_tile_warp_col_idx][thread_tile_x_vector_idx * NUM_VECTOR_UNITS]) = 
                *reinterpret_cast<int4*>(&B_thread_block_tile[B_thread_tile_row_idx][B_thread_tile_col_idx + thread_tile_x_vector_idx * NUM_VECTOR_UNITS]);
            }
        }
     }

    for(size_t thread_tile_warp_row_idx = 0; thread_tile_warp_row_idx < NUM_THREAD_TILES_PER_WARP_Y; ++thread_tile_warp_row_idx)
    {
        for(size_t thread_tile_warp_col_idx = 0; thread_tile_warp_col_idx < NUM_THREAD_TILES_PER_WARP_X; ++thread_tile_warp_col_idx)
        {
            for(size_t thread_tile_y_vector = 0; thread_tile_y_vector < VECTORIZED_THREAD_TILE_SIZE_Y; ++thread_tile_y_vector)
            {
                for(size_t thread_tile_x_vector = 0; thread_tile_x_vector < VECTORIZED_THREAD_TILE_SIZE_X; ++thread_tile_x_vector)
                {
                    C_thread_results[thread_tile_warp_row_idx][thread_tile_warp_col_idx][thread_tile_y_vector][thread_tile_x_vector] += A_vals[thread_tile_warp_row_idx][thread_tile_y_vector] * B_vals[thread_tile_warp_col_idx][thread_tile_x_vector];
                }
            }
        }
        __syncthreads();
    }

    for(size_t thread_tile_warp_row_idx = 0; thread_tile_warp_row_idx < NUM_THREAD_TILES_PER_WARP_Y; ++thread_tile_warp_row_idx)
    {
        for(size_t thread_tile_warp_col_idx = 0; thread_tile_warp_row_idx <NUM_THREAD_TILES_PER_WARP_X; ++thread_tile_warp_row_idx)
        {
            for(size_t thread_tile_y_idx = 0; thread_tile_y_idx < VECTORIZED_THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
            {
                for(size_t thread_tile_x_idx = 0; thread_tile_x_idx < VECTORIZED_THREAD_TILE_SIZE_Y; ++thread_tile_x_idx)
                {
                    size_t const C_row_idx = blockIdx.y * BLOCK_TILE_Y +
                                             warp_row_idx * WARP_TILE_SIZE_Y +
                                             thread_tile_warp_row_idx * (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                                             thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y + thread_tile_y_idx;
                    size_t const C_col_idx = blockIdx.x * BLOCK_TILE_X +
                                             warp_row_idx *  WARP_TILE_SIZE_X +
                                             thread_tile_warp_col_idx * (WARP_TILE_SIZE_X /NUM_THREAD_TILES_PER_WARP_X) +
                                             thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X + 
                                             thread_tile_x_idx * NUM_VECTOR_UNITS;

                    if(C_row_idx < m &&C_col_idx < n)
                    {
                        int4 C_vals = *reinterpret_cast<int4 const*>(&C[C_row_idx * ldc + C_col_idx]) ;

                        for(size_t i = 0; i < NUM_VECTOR_UNITS; ++i)
                        {
                            reinterpret_cast<T*>(&C_vals)[i] = alpha * C_thread_results[thread_tile_warp_row_idx][thread_tile_warp_col_idx][thread_tile_y_idx][thread_tile_x_idx * NUM_VECTOR_UNITS + i] +
                                                                beta * reinterpret_cast<T const*>(&C_vals)[i];
                        }
                        *reinterpret_cast<int4*>(&C[C_row_idx * ldc + C_col_idx]) = C_vals;
                    }
                                

                }
            }
        }
    }
    

}
template<typename T>
void launch_kernel_gemm_v06(size_t m, size_t n , size_t k, T alpha, T const* A, size_t lda,
                             T const* B, size_t ldb, T beta, T* C, size_t ldc, cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_X = 128;
    constexpr unsigned int BLOCK_TILE_Y = 128;
    constexpr unsigned int BLOCK_TILE_K = 16;

    constexpr unsigned int WARP_TILE_SIZE_X = 32;
    constexpr unsigned int WARP_TILE_SIZE_Y = 64;

    constexpr unsigned int NUM_WARPS_X = BLOCK_TILE_X / WARP_TILE_SIZE_X;
    constexpr unsigned int NUM_WARPS_Y = BLOCK_TILE_Y / WARP_TILE_SIZE_Y;

    static_assert(BLOCK_TILE_X % WARP_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_Y % WARP_TILE_SIZE_Y == 0);

    constexpr unsigned int THREAD_TILE_SIZE_X = 8;
    constexpr unsigned int THREAD_TILE_SIZE_Y = 8;

    constexpr unsigned int NUM_THREADS_PER_WARP_X = WARP_TILE_SIZE_X / THREAD_TILE_SIZE_X;
    constexpr unsigned int NUM_THREADS_PER_WARP_Y = WARP_TILE_SIZE_Y / THREAD_TILE_SIZE_Y;

    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == WARP_SIZE);

    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);


    constexpr unsigned int NUM_THREADS_X = NUM_WARPS_X * NUM_THREADS_PER_WARP_X;
    constexpr unsigned int NUM_THREADS_Y = NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y;

    //constexpr unsigned int NUM_THREADS_PER_BLOCK = BLOCK_TILE_X * BLOCK_TILE_Y /(WARP_TILE_SIZE_X * WARP_TILE_SIZE_Y) * WARP_SIZE;
    constexpr unsigned int NUM_THREADS_PER_BLOCK = NUM_THREADS_X * NUM_THREADS_Y;
    
    dim3 block_dim = {NUM_THREADS_PER_BLOCK,1,1};
    dim3 grid_dim = {(static_cast<unsigned int>(n) + BLOCK_TILE_X -1) / BLOCK_TILE_X,
                    (static_cast<unsigned int>(m) + BLOCK_TILE_Y -1) / BLOCK_TILE_Y,
                    1};

    gemm_v06<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, 
            THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y, NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y><<<grid_dim, block_dim, 0, stream>>>(m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);

     CHECK_LAST_ERROR();

}

#define STATIC_ASSERT(a,b) static_assert( (a % b) == 0)

template<typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y, size_t BLOCK_TILE_K, size_t BLOCK_TILE_SKEW_X, size_t BLOCK_TILE_SKEW_Y,
        size_t WARP_TILE_X, size_t WARP_TILE_Y,size_t WMMA_TILE_SIZE_X, size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_v07_vectorized(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda, 
                                    T const* B, size_t ldb, T beta, T* C, size_t ldc)                               
{
    static_assert((BLOCK_TILE_X % WARP_TILE_X) == 0);
    static_assert((BLOCK_TILE_Y % WARP_TILE_Y) == 0);

    constexpr size_t NUM_WARP_X = BLOCK_TILE_X / WARP_TILE_X;

    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_K][BLOCK_TILE_Y + BLOCK_TILE_SKEW_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_K][BLOCK_TILE_X + BLOCK_TILE_SKEW_X];

    STATIC_ASSERT(WARP_TILE_X, WMMA_TILE_SIZE_X);
    STATIC_ASSERT(WARP_TILE_Y, WMMA_TILE_SIZE_Y);
    STATIC_ASSERT(BLOCK_TILE_K,WMMA_TILE_SIZE_K);

    //，X Y 执行paraller
    constexpr size_t NUM_WMMA_TILES_X = WARP_TILE_X / WMMA_TILE_SIZE_X;
    constexpr size_t NUM_WMMA_TILES_Y = WARP_TILE_Y / WMMA_TILE_SIZE_Y;
    //k方向执行reduction
    constexpr size_t NUM_WMMA_TILES_K = BLOCK_TILE_K / WMMA_TILE_SIZE_K; 

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X,
                            WMMA_TILE_SIZE_K, T, nvcuda::wmma::col_major>  a_frags[NUM_WMMA_TILES_Y];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y,
     WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K,T, nvcuda::wmma::row_major> b_frags[NUM_WMMA_TILES_X];


    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X, 
                            WMMA_TILE_SIZE_K, T> acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_X,
                             WMMA_TILE_SIZE_K, T> c_frag;
    

    //nvcuda::wmma::fill_fragment(c_frag, 0U);
#pragma unroll
    for(size_t wmma_tile_row_idx = 0; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
    {
        for(size_t wmma_tile_col_idx = 0; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_row_idx)
        {
            nvcuda::wmma::fill_fragment(acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],static_cast<T>(0));
        }
    }

    size_t const thread_linear_idx = threadIdx.x + threadIdx.y * blockDim.x;

    size_t const warp_linear_idx = thread_linear_idx / WARP_SIZE;

    size_t const warp_row_idx = warp_linear_idx / NUM_WARP_X;
    size_t const warp_col_idx = warp_linear_idx % NUM_WARP_X;

    //执行规约的次数，也是从global -> shared 的次数  可paraller的地方
    size_t const num_thread_block_tiles = (k + BLOCK_TILE_K -1) / BLOCK_TILE_K;

    for(size_t num_thread_block_tiles_idx = 0; num_thread_block_tiles_idx < num_thread_block_tiles; ++num_thread_block_tiles_idx)
    {
        load_data_to_shared_memory_transposed_vectorized<T, BLOCK_TILE_X, BLOCK_TILE_Y, BLOCK_TILE_K, 
        NUM_THREADS, BLOCK_TILE_SKEW_X, BLOCK_TILE_SKEW_Y>(A, lda, B, ldb, A_thread_block_tile_transposed,
        B_thread_block_tile, num_thread_block_tiles_idx, thread_linear_idx, m, n, k);
        __syncthreads();
    } 
#pragma unroll
     for(size_t k_i = 0; k_i < NUM_WMMA_TILES_K; ++k_i)
     {
#pragma unroll        
        for(size_t wmma_tile_row_idx = 0; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
        {
            nvcuda::wmma::load_matrix_sync(a_frags[wmma_tile_row_idx], 
                                         &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K][warp_row_idx * WARP_TILE_Y + wmma_tile_row_idx * WMMA_TILE_SIZE_Y],
                                        BLOCK_TILE_Y + BLOCK_TILE_SKEW_Y);

            for(size_t wmma_tile_col_idx = 0; wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
            {
                nvcuda::wmma::load_matrix_sync(b_frags[wmma_tile_col_idx],
                                           &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K][warp_col_idx * WARP_TILE_X + wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                                           BLOCK_TILE_X + BLOCK_TILE_SKEW_X);

                nvcuda::wmma::mma_sync(acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                                    a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                                    acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
            }
        }
     }  
     __syncthreads();  

#pragma unroll 
    for(size_t wmma_tile_row_idx = 0; wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
    {
#pragma unroll
        for(size_t wmma_tile_col_idx = 0; wmma_tile_col_idx <NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
        {
            //I * n +J ；也可以使用 J * M + I 但是要符合连续存储读写的要求，
            //这里是在共享内存中给c_frag 开一片内存
            nvcuda::wmma::load_matrix_sync(c_frag,&C[(blockIdx.y * BLOCK_TILE_Y + 
                                                        warp_row_idx * WARP_TILE_Y +
                                                        wmma_tile_row_idx * NUM_WMMA_TILES_Y) * n + 
                                                        blockIdx.x * BLOCK_TILE_X +
                                                        warp_col_idx * WARP_TILE_X +
                                                        wmma_tile_col_idx * NUM_WMMA_TILES_X],
                                                        n, 
                                                         nvcuda::wmma::mem_row_major);
            for(size_t i = 0; i < c_frag.num_elements; ++i)
            {
                c_frag.x[i] = alpha * acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] + 
                              beta * c_frag.x[i];
            }
            //st 和ld的区别
            nvcuda::wmma::store_matrix_sync(&C[(blockIdx.y * BLOCK_TILE_Y +
                                            warp_row_idx * WARP_TILE_Y +
                                            wmma_tile_row_idx * NUM_WMMA_TILES_Y) * n + 
                                            blockIdx.x * BLOCK_TILE_X +
                                            warp_col_idx * WARP_TILE_X +
                                            wmma_tile_col_idx * NUM_WMMA_TILES_X],
                                            c_frag,
                                            n, 
                                            nvcuda::wmma::mem_row_major);
        }
    }
    
}
#define DIV_CEIL(a,b) (a + b -1) / b
template<typename T>
void launch_gemm_kernel_v07_vectorized(int m, int n, int k, T alpha,
                                     T const* A, size_t lda, T const* B, size_t ldb,
                                     T beta, T* C, size_t ldc, cudaStream_t stream)
{
    constexpr unsigned int BLOCK_TILE_X = 128;
    constexpr unsigned int BLOCK_TILE_Y = 128;
    constexpr unsigned int BLOCK_TILE_K = 16;

    constexpr unsigned int BLOCK_TILE_SKEW_X = 16;
    constexpr unsigned int BLOCK_TILE_SKEW_Y = 16;

    constexpr unsigned int WARP_TILE_X = 16;
    constexpr unsigned int WARP_TILE_Y = 16;

    STATIC_ASSERT(BLOCK_TILE_X, WARP_TILE_X);
    STATIC_ASSERT(BLOCK_TILE_Y, WARP_TILE_Y);
    //WARP 中矩阵的布局
    constexpr unsigned int NUM_WARPS_X = BLOCK_TILE_X / WARP_TILE_X;
    constexpr unsigned int NUM_WARPS_Y = BLOCK_TILE_Y / WARP_TILE_Y;

    //tensor core的大小
    constexpr unsigned int wmma_tile_size_x = 16;
    constexpr unsigned int wmma_tile_size_y = 16;
    constexpr unsigned int wmma_tile_size_k = 16;

    constexpr unsigned int NUM_THREADS_PER_BLOCK = NUM_WARPS_X * NUM_WARPS_Y * WARP_SIZE;

    dim3 block_dim = {NUM_THREADS_PER_BLOCK, 1, 1};
    dim3 grid_dim ={ DIV_CEIL(n,BLOCK_TILE_Y), 
                    DIV_CEIL(m,BLOCK_TILE_X),
                    1};
   gemm_v07_vectorized<T, BLOCK_TILE_X, BLOCK_TILE_Y,
    BLOCK_TILE_K, BLOCK_TILE_SKEW_X, BLOCK_TILE_SKEW_Y, WARP_TILE_X, WARP_TILE_Y,
    wmma_tile_size_x, wmma_tile_size_y, wmma_tile_size_k, NUM_THREADS_PER_BLOCK>
    <<<grid_dim, block_dim, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 


}



int main()
{
    int m = 256,n = 256, k = 32;
    half alpha =1;//beat = 0;

    half  beat1 = 0;
    half* A = NULL;
    half* B = NULL;
    half* C = NULL; 
    
    size_t lda = 256;
    size_t ldb = 256;
    size_t ldc = 256;
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_gemm_kernel_v07_vectorized<half>(m,n,k,alpha, A, lda,B,ldb,beat1,C,ldc,stream);
    //launch_gemm_kernel_v01<int>(m,n,k,alpha, A, alpha,B,alpha,beat,C,alpha,stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaStreamCreate(&stream);
    //launch_gemm_kernel_v02<int>(m,n,k,alpha, A, alpha,B,alpha1,beat1,C,alpha,stream);
    return 0;
}

