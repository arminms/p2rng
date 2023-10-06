#ifndef _P2RNG_ALGORITHM_GENERATE_HPP_
#define _P2RNG_ALGORITHM_GENERATE_HPP_

//----------------------------------------------------------------------------//
// oneAPI

#if defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)

namespace p2rng::oneapi {

template <typename ForwardIt, typename Generator>
inline auto generate
(   ForwardIt s
,   ForwardIt e
,   Generator g
,   sycl::queue q = sycl::queue()
)-> sycl::event
{   auto n{std::distance(s, e)};
    auto event = q.submit
    (   [&](sycl::handler& h)
        {   const decltype(n) threads_per_block{256};
            const decltype(n) blocks_per_grid{n / threads_per_block + 1};
            const decltype(n) job_size{blocks_per_grid * threads_per_block};
            sycl::buffer buf_out = s.get_buffer();
            sycl::accessor oa(buf_out, h, sycl::write_only);
            h.parallel_for
            (   sycl::nd_range<1>
                (   sycl::range<1>(job_size)
                ,   sycl::range<1>(threads_per_block)
                )
            ,   [=](sycl::nd_item<1> itm)
                {   auto tlg = g;   // making a a thread local copy
                    auto idx
                    {   itm.get_group(0)
                    *   itm.get_local_range(0)
                    +   itm.get_local_id(0)
                    };
                    if (idx < n)
                    {   tlg.discard(idx);
                        oa[idx] = tlg();
                    }
                }
            );
        }
    );
    return event;
}

} // end p2rng::oneapi namespace

//----------------------------------------------------------------------------//
// CUDA

#elif defined(__CUDACC__)

namespace p2rng::cuda {

template<typename T, typename SizeT, typename GeneratorT>
__global__ void block_splitting
(   T* out
,   SizeT n
,   GeneratorT g
)
{   auto idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < n)
    {   g.discard(idx);
        out[idx] = g();
    }
}

template <typename ForwardIt, typename Generator>
inline void generate
(   ForwardIt s
,   ForwardIt e
,   Generator g
)
{   auto n{std::distance(s, e)};
    const decltype(n) threads_per_block{256};
    decltype(n) blocks_per_grid{n / threads_per_block + 1}; 
    block_splitting<<<blocks_per_grid, threads_per_block>>>
    (   thrust::raw_pointer_cast(&s[0])
    ,   n
    ,   g
    );
}

} // end p2rng::cuda namespace

//----------------------------------------------------------------------------//
// ROCm

#elif defined(__HIP_PLATFORM_AMD__)

namespace p2rng::rocm {

template<typename T, typename SizeT, typename GeneratorT>
__global__ void block_splitting
(   T* out
,   SizeT n
,   GeneratorT g
)
{   auto idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx < n)
    {   g.discard(idx);
        out[idx] = g();
    }
}

template <typename ForwardIt, typename Generator>
inline void generate
(   ForwardIt s
,   ForwardIt e
,   Generator g
)
{   auto n{std::distance(s, e)};
    const decltype(n) threads_per_block{256};
    decltype(n) blocks_per_grid{n / threads_per_block + 1}; 
    block_splitting<<<blocks_per_grid, threads_per_block>>>
    (   thrust::raw_pointer_cast(&s[0])
    ,   n
    ,   g
    );
}

} // end p2rng::rocm namespace

#else

#   include <omp.h>

namespace p2rng {

//----------------------------------------------------------------------------//
// OpenMP

template <typename ForwardIt, typename Generator>
inline void generate
(   ForwardIt s
,   ForwardIt e
,   Generator g
)
{
    #pragma omp parallel
    {   auto n{std::distance(s, e)};
        auto tidx{omp_get_thread_num()};
        auto size{omp_get_num_threads()};
        auto first{tidx * n / size};
        auto last{(tidx + 1) * n / size};
        auto tlg = g;   // making a thread local copy
        tlg.discard(first);
        for (auto i{first}; i < last; ++i)
            s[i] = tlg();
    }
}

} // end p2rng namespace

#endif  //__INTEL_LLVM_COMPILER && SYCL_LANGUAGE_VERSION

#endif  //_P2RNG_ALGORITHM_GENERATE_HPP_