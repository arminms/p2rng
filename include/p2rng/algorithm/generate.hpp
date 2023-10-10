//
// Copyright (c) 2023 Armin Sobhani (https://arminsobhani.ca)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
#ifndef _P2RNG_ALGORITHM_GENERATE_HPP_
#define _P2RNG_ALGORITHM_GENERATE_HPP_

#if !(defined(SYCL_LANGUAGE_VERSION) || defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__))
#   include <omp.h>
#endif
namespace p2rng {

/**
 * === oneAPI ==================================================================
 */

#if defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)

/**
 *  @brief Assigns @a n random numbers using SYCL device, generated by given
 *  function object @a g.
 *
 *  The random numbers are assigned to the first @a n elements in the range
 *  beginning at \a out, if \a n > 0. Does nothing otherwise. @a g must be
 *  either a random number engine or a bind object formed from a distribution
 *  and an engine returned by \a p2rng::bind(). Lambdas are not supported.
 *  @ingroup mutating_algorithms
 *  @tparam OutputIt iterator type for @a out
 *  @tparam Size type for @a n
 *  @tparam Generator generator type for @a g
 *  @param  out the beginning of the range of random numbers to generate
 *  @param  n   number of random numbers to generate
 *  @param  g   generator function object. Only a random number engine or a bind
 *              object returned by \a p2rng::bind() are valid.
 *  @param  q   optional sycl::queue object to submit the command
 *  @return sycl::event object of the submitted command
 */
template <typename OutputIt, typename Size, typename Generator>
inline auto generate_n
(   OutputIt out
,   Size n
,   Generator g
,   sycl::queue q = sycl::queue()
)-> sycl::event
{   auto event = q.submit
    (   [&](sycl::handler& h)
        {   const Size threads_per_block{256};
            const Size blocks_per_grid{n / threads_per_block + 1};
            const Size job_size{blocks_per_grid * threads_per_block};
            sycl::buffer buf_out = out.get_buffer();
            sycl::accessor oa(buf_out, h, sycl::write_only);
            h.parallel_for
            (   sycl::nd_range<1>
                (   sycl::range<1>(job_size)
                ,   sycl::range<1>(threads_per_block)
                )
            ,   [=](sycl::nd_item<1> itm)
                {   auto tlg = g;   // make a thread local copy
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

/**
 *  @brief Assigns to each elements in the range @p [first,last)
 *  a random number generated by given function object @a g on the GPU.
 *
 *  @a g must be either a random number engine or a bind object formed from a
 *  distribution and an engine returned by \a p2rng::bind(). Lambdas are not
 *  supported.
 *  @ingroup mutating_algorithms
 *  @tparam ForwardIt iterator type for @a first and @a last
 *  @tparam Size type for @a n
 *  @tparam Generator generator type for @a g
 *  @param  first the beginning of the range of random numbers to generate
 *  @param  last  the end of the range of random numbers to generate
 *  @param  g     generator function object. Only a random number engine or a
 *                bind object returned by \a p2rng::bind() are valid.
 *  @param  q     optional sycl::queue object to submit the command
 *  @return sycl::event object of the submitted command
 */
template <typename ForwardIt, typename Generator>
inline auto generate
(   ForwardIt first
,   ForwardIt last
,   Generator g
,   sycl::queue q = sycl::queue()
)-> sycl::event
{   auto n{std::distance(first, last)};
    return p2rng::generate_n(first, n, g, q);
}

/**
 * === CUDA / ROCm =============================================================
 */

#elif defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)

/**
 * device kernel
 */

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

/**
 *  @brief Assigns @a n random numbers using GPU, generated by given function
 *  object @a g.
 *
 *  The random numbers are assigned to the first @a n elements in the range
 *  beginning at \a out, if \a n > 0. Does nothing otherwise. @a g must be
 *  either a random number engine or a bind object formed from a distribution
 *  and an engine returned by \a p2rng::bind(). Lambdas are not supported.
 *  @ingroup mutating_algorithms
 *  @tparam OutputIt iterator type for @a out
 *  @tparam Size type for @a n
 *  @tparam Generator generator type for @a g
 *  @param  out the beginning of the range of random numbers to generate
 *  @param  n   number of random numbers to generate
 *  @param  g   generator function object. Only a random number engine or a bind
 *              object returned by \a p2rng::bind() are valid.
 *  @return Iterator one past the last random number if @a n > 0, @a out
 *          otherwise.
 */
template <typename OutputIt, typename Size, typename Generator>
inline OutputIt generate_n
(   OutputIt out
,   Size n
,   Generator g
)
{   const Size threads_per_block{256};
    Size blocks_per_grid{n / threads_per_block + 1}; 
    block_splitting<<<blocks_per_grid, threads_per_block>>>
    (   thrust::raw_pointer_cast(&out[0])
    ,   n
    ,   g
    );
    std::advance(out, n);
    return out;
}

/**
 *  @brief Assigns to each elements in the range @p [first,last)
 *  a random number generated by given function object @a g on the GPU.
 *
 *  @a g must be either a random number engine or a bind object formed from a
 *  distribution and an engine returned by \a p2rng::bind(). Lambdas are not
 *  supported.
 *  @ingroup mutating_algorithms
 *  @tparam ForwardIt iterator type for @a first and @a last
 *  @tparam Size type for @a n
 *  @tparam Generator generator type for @a g
 *  @param  first the beginning of the range of random numbers to generate
 *  @param  last  the end of the range of random numbers to generate
 *  @param  g     generator function object. Only a random number engine or a
 *                bind object returned by \a p2rng::bind() are valid.
 *  @return none
 */
template <typename ForwardIt, typename Generator>
inline void generate
(   ForwardIt first
,   ForwardIt last
,   Generator g
)
{   auto n{std::distance(first, last)};
    p2rng::generate_n(first, n, g);
}

/**
 * === OpenMP ==================================================================
 */

#else

/**
 *  @brief Assigns @a n random numbers in parallel, generated by given function
 *  object @a g.
 *
 *  The random numbers are assigned to the first @a n elements in the range
 *  beginning at \a out, if \a n > 0. Does nothing otherwise. @a g must be
 *  either a random number engine or a bind object formed from a distribution
 *  and an engine returned by \a p2rng::bind(). Lambdas are not supported.
 *  @ingroup mutating_algorithms
 *  @tparam OutputIt iterator type for @a out
 *  @tparam Size type for @a n
 *  @tparam Generator generator type for @a g
 *  @param  out the beginning of the range of random numbers to generate
 *  @param  n   number of random numbers to generate
 *  @param  g   generator function object. Only a random number engine or a bind
 *              object returned by \a p2rng::bind() are valid.
 *  @return Iterator one past the last random number if @a n > 0, @a out
 *          otherwise.
 */
template <typename OutputIt, typename Size, typename Generator>
inline OutputIt generate_n
(   OutputIt out
,   Size n
,   Generator g
)
{
    #pragma omp parallel
    {   auto tidx{omp_get_thread_num()};
        auto size{omp_get_num_threads()};
        Size first{tidx * n / size};
        Size last{(tidx + 1) * n / size};
        auto tlg = g;   // make a thread local copy
        tlg.discard(first);
        for (auto i{first}; i < last; ++i)
            out[i] = tlg();
    }
    std::advance(out, n);
    return out;
}

/**
 *  @brief Assigns in parallel to each elements in the range @p [first,last)
 *  a random number generated by given function object @a g.
 *
 *  @a g must be either a random number engine or a bind object formed from a
 *  distribution and an engine returned by \a p2rng::bind(). Lambdas are not
 *  supported.
 *  @ingroup mutating_algorithms
 *  @tparam ForwardIt iterator type for @a first and @a last
 *  @tparam Size type for @a n
 *  @tparam Generator generator type for @a g
 *  @param  first the beginning of the range of random numbers to generate
 *  @param  last  the end of the range of random numbers to generate
 *  @param  g     generator function object. Only a random number engine or a
 *                bind object returned by \a p2rng::bind() are valid.
 *  @return none
 */
template <typename ForwardIt, typename Generator>
inline void generate
(   ForwardIt first
,   ForwardIt last
,   Generator g
)
{   auto n{std::distance(first, last)};
    p2rng::generate_n(first, n, g);
}

#endif  //__INTEL_LLVM_COMPILER && SYCL_LANGUAGE_VERSION

} // end p2rng namespace

#endif  //_P2RNG_ALGORITHM_GENERATE_HPP_