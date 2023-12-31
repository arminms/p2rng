[![Build and Test (Linux/macOS/Windows)](https://github.com/arminms/p2rng/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/arminms/p2rng/actions/workflows/cmake-multi-platform.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# p2rng 
`p2rng` (`P`arallel `P`seudo `R`andom `N`umber `G`enerator) is a modern header-only C++
library for parallel algorithmic (pseudo) random number generation supporting
[`OpenMP`](https://www.openmp.org/), [`CUDA`](https://developer.nvidia.com/cuda-zone), [`ROCm`](https://www.amd.com/en/graphics/servers-solutions-rocm) and
[`oneAPI`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html).

`p2rng` provides alternatives to STL `generate()` family of algorithms that exclusively designed for parallel random number generation on CPUs and GPUs. Unlike C++17 parallel version of [`std::generate()`](https://en.cppreference.com/w/cpp/algorithm/generate) and [`std::generate_n()`](https://en.cppreference.com/w/cpp/algorithm/generate_n) that cannot be readily used for random number generation, `p2rng::generate()` and `p2rng::generate_n()` can do it hassle-free with almost the
same interface.

One important feature of `generate()` algorithms provided by `p2rng` is that they play fair: using the same seed and distribution, you can get the same sequence of random numbers on all supported platforms regardless of the number of parallel threads. CUDA, ROCm and oneAPI provide their own parallel random number libraries: [cuRAND](https://docs.nvidia.com/cuda/curand/index.html), [rocRAND](https://rocm.docs.amd.com/projects/rocRAND) and [oneMKL](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/overview.html). Aside from the fact that their interface is not compatible with STL, it's almost impossible to get the the same sequence of random numbers using them.

## Table of contents
- [Features](#features)
- [Building from source](#building-from-source)
- [Running unit tests](#running-unit-tests)
- [Running benchmarks](#running-benchmarks)
- [Using *p2rng*](#using-p2rng)
  - [Example](#example)
  - [YouTube Video](#youtube-video)

## Features
- Multiplatform
  - Linux
  - macOS
  - Windows 10/11
- Support four target APIs
  - [`CUDA`](https://developer.nvidia.com/cuda-zone)
  - [`oneAPI`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
  - [`OpenMP`](https://www.openmp.org/)
  - [`ROCm`](https://www.amd.com/en/graphics/servers-solutions-rocm)
- Provide parallel versions of STL’s
[`std::generate()`](https://en.cppreference.com/w/cpp/algorithm/generate) and [`std::generate_n()`](https://en.cppreference.com/w/cpp/algorithm/generate_n) algorithms with the same interface 
- Play fair on all supported platforms
- Included engines:
  - [PCG Family](https://www.pcg-random.org/)
- Include all 32 distributions provided by [TRNG](https://github.com/rabauke/trng4) library
- Support [`CMake`](https://cmake.org/) for building and auto configuration
- Include unit tests using [`Catch2`](https://github.com/catchorg/Catch2)
- Include benchmarks using [`Google Benchmark`](https://github.com/google/benchmark)

## Building from source
You need:
- C++ compiler supporting the C++17 standard (e.g. `gcc` 9.3)
- [`CMake`](https://cmake.org/) version 3.21 or higher.

And the following optional third-party libraries:
* [Catch2](https://github.com/catchorg/Catch2) v3.1 or higher for unit testing
* [Google Benchmark](https://github.com/google/benchmark) for benchmarks

The `CMake` script configured in a way that if it cannot find the optional third-party libraries it tries to fetch and build them automatically. So, there is no need to do anything if they are missing but you need an internet connection for that to work.

On [the Alliance](https://alliancecan.ca/) clusters, you can activate the above environment by the following module command:
```shell
module load cmake googlebenchmark catch2
```
Once you have all the requirements you can build and install it using the
following commands:
```shell
git clone https://github.com/arminms/p2rng.git
cd p2rng
cmake -S . -B build
cmake --build build -j
sudo cmake --install build
```
## Running unit tests
```shell
cd build
ctest
```
## Running benchmarks
```shell
cd build
perf/benchmarks --benchmark_counters_tabular=true
```
## Using `p2rng`
Ideally you should be using `p2rng` through its CMake integration. `CMake` build
of `p2rng` exports four (namespaced) targets:
- `p2rng::cuda`
- `p2rng::oneapi`
- `p2rng::openmp`
- `p2rng::rocm`

Linking against them adds the proper include paths and links your target with
proper libraries depending on the API. This means that if `p2rng` has been installed on the system, it should be enough to do:
```cmake
find_package(p2rng CONFIG COMPONENTS openmp cuda)

# link test.cpp with p2rng using OpenMP API
add_executable(test_openmp test.cpp)
target_link_libraries(test_openmp PRIVATE p2rng::openmp)

# link test.cu with p2rng using CUDA API
add_executable(test_cuda test.cu)
target_link_libraries(test_cuda PRIVATE p2rng::cuda)
```

Another possibility is to check if `p2rng` is installed and if not use
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html):

```cmake
# include the module
include(FetchContent)

# first check if p2rng is already installed
find_package(p2rng CONFIG COMPONENTS oneapi)

# if not, try to fetch and make it available
if(NOT p2rng_FOUND)
  message(STATUS "Fetching p2rng library...")
  FetchContent_Declare(
    p2rng
    GIT_REPOSITORY https://github.com/arminms/p2rng.git
    GIT_TAG main
  )
  # setting required p2rng components
  set(P2RNG_COMPONENTS oneapi CACHE STRING "Required components")
  FetchContent_MakeAvailable(p2rng)
endif()

# link test.cpp with p2rng using oneapi as API
add_executable(test_oneapi test.cpp)
target_link_libraries(test_oneapi PRIVATE p2rng::oneapi)
```
### Example
You can find a complete example of the above approach in the [`example`](example/) folder along with build instructions.

### YouTube Video
There is a recorded video about `p2rng` on [SHARCNET YouTube Channel](https://youtube.sharcnet.ca):

* [p2rng – A C++ Parallel Random Number Generator Library for the Masses](https://youtu.be/nyOH8nvHBss)

You can find PDF of the slides in the [doc](./doc) folder. All the codes shown in the above video are also included in the [doc/codes](./doc/codes) folder along with a CMake script to build them.