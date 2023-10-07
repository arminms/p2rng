[![Build and Test](https://github.com/arminms/p2rng/actions/workflows/cmake.yml/badge.svg)](https://github.com/arminms/p2rng/actions/workflows/cmake.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# p2rng 
`p2rng` (Parallel Pseudo Random Number Generator) is a modern header-only C++
library for parallel algorithmic (pseudo) random number generation supporting
[`OpenMP`](https://www.openmp.org/), [`CUDA`](https://developer.nvidia.com/cuda-zone),
[`ROCm`](https://www.amd.com/en/graphics/servers-solutions-rocm) and
[`oneAPI`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html).

## Table of contents
- [Features](#features)
- [Building from source](#building-from-source)
- [Running unit tests](#running-unit-tests)
- [Running benchmarks](#running-benchmarks)
- [Using *p2rng*](#using-p2rng)

## Features
- Support four target APIs
    - [`CUDA`](https://developer.nvidia.com/cuda-zone)
    - [`oneAPI`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
    - [`OpenMP`](https://www.openmp.org/)
    - [`ROCm`](https://www.amd.com/en/graphics/servers-solutions-rocm)
- Provide parallel versions of STL’s
[`std::generate()`](https://en.cppreference.com/w/cpp/algorithm/generate) and [`std::generate_n()`](https://en.cppreference.com/w/cpp/algorithm/generate_n) algorithms with the same interface 
- Play fair on all supported platforms (using the same seed and distribution you
 always get the same sequence of random numbers)
- Included engines:
    - [PCG Family](https://www.pcg-random.org/)
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
find_package(p2rng REQUIRED)

# link test1 with p2rng using OpenMP API
add_executable(test1 test1.cpp)
target_link_libraries(test1 PRIVATE p2rng::openmp)

# link test2 with p2rng using CUDA API
add_executable(test2 test2.cpp)
target_link_libraries(test2 PRIVATE p2rng::cuda)
```

Another possibility is to check if `p2rng` is installed and if not use
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html):

```cmake
find_package(p2rng)
if(NOT p2rng_FOUND)
  message(STATUS "Fetching p2rng library...")
  FetchContent_Declare(
    p2rng
    GIT_REPOSITORY https://github.com/arminms/p2rng.git
    GIT_TAG        v0.1.0
  )
  FetchContent_MakeAvailable(p2rng)
endif()

# link test with p2rng using oneapi API
add_executable(test test.cpp)
target_link_libraries(test PRIVATE p2rng::oneapi)

```
