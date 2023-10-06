#ifndef _P2RNG_DEVICE_HPP_
#define _P2RNG_DEVICE_HPP_

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
#   define P2RNG_DEVICE_CODE __device__ __host__
// #elif defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)
// #   define P2RNG_DEVICE_CODE extern SYCL_EXTERNAL
#else
#   define P2RNG_DEVICE_CODE
#endif

#endif  //_P2RNG_DEVICE_HPP_