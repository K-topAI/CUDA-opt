#inlcue<cuda/barrier>
#include<cuda/ptx>
using barrier=cuda::barrier<cuda::thread_scope_block>;
namespace ptx=cuda::ptx;
static constexpr size_t buf_len=1024;
__global__ void add_one_kernel(int* data,size_t offset)
{
    __shared__ alignas(16) int smem_data[buf_len];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if(threadIdx.x == 0)
    {
        init(&bar, blockDim.x);
        ptx::fence_proxy_async(ptx::shared);
    }
    __syncthreads();

    if(threadIdx.x == 0){
        cuda::memcpy_async{
            smem_data,
            data + offset,
            cuda::aligned_size_t<16>(sizeof(smem_data)),
            bar,
        }
    }

    barrier::arrival_token token=bar.arrive();

    bar.wait(std::move(token));

    for(int i=threadIdx.x; i<buf_len; i+=blockDim.x)
    {
        smem_data[i] += 1;
    }

    ptx::fence_proxy_async(ptx::shared);

    __syncthreads();

    if(threadIdx.x == 0){
        ptx::cp_async_bulk(
            ptx::space_global;
            ptx::space_shared;
            data + offset,smem_data,size_of(smem_data)
        );

        ptx::cp_async_bulk_commit_group();

        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }


}
