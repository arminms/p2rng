# Example
This folder contains a complete example of how to use `p2rng` including the `CMakeLists.txt` file. There are three source files for different APIs that doing the exact same thing:
```shell
└── example
    ├── CMakeLists.txt
    ├── README.md
    ├── rand100.cu
    ├── rand100_oneapi.cpp
    └── rand100_openmp.cpp
```
## CMake script (`CMakeLists.txt`)
The CMake script always builds the `OpenMP` version. Depending on availability of other APIs, it tries to build them as well and ignores otherwise. That's why `cuda`, `oneapi` and `rocm` are listed as `OPTIONAL_COMPONENTS` in `find_package(p2rng)` command. It uses the same source (i.e. `rand100.cu`) for both `CUDA` and `ROCm` versions:
```cmake
cmake_minimum_required(VERSION 3.21...3.26)

project(rand100 CXX)

## include necessary modules
#
include(FetchContent)
include(CheckLanguage)

## find p2rng, if not installed fetch it...
#
find_package(p2rng CONFIG
  COMPONENTS openmp
  OPTIONAL_COMPONENTS cuda oneapi rocm
)
if(NOT p2rng_FOUND)
  message(STATUS "Fetching p2rng library...")
  FetchContent_Declare(
    p2rng
    GIT_REPOSITORY https://github.com/arminms/p2rng.git
    GIT_TAG main
  )
  FetchContent_MakeAvailable(p2rng)
endif()

## build OpenMP version by default
#
add_executable(rand100_openmp rand100_openmp.cpp)
target_link_libraries(rand100_openmp PRIVATE p2rng::openmp)

## build CUDA version if it's available
#
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(rand100_cuda rand100.cu)
  target_link_libraries(rand100_cuda PRIVATE p2rng::cuda)
endif()

## build ROCm version if it's available
#
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  add_executable(rand100_rocm rand100.cu)
  target_link_libraries(rand100_rocm PRIVATE p2rng::rocm)
endif()

## build oneAPI version if it's available
#
find_package(IntelDPCPP CONFIG)
if (IntelDPCPP_FOUND)
  add_executable(rand100_oneapi rand100_oneapi.cpp)
  target_link_libraries(rand100_oneapi PRIVATE p2rng::oneapi)
endif()
```
To build for `OpenMP` and `CUDA`, you can use the following commands inside the folder:
```shell
cmake -S . -B build
cmake --build build -j
```
Here's the output for `OpenMP` version but you get the exact same output using other APIs:
```
$ build/rand100_openmp

 82 82 42 72 75 37 38 55 88 48
 68 80 78 32 21 13 52 18 53 55
 73 58 80 52 78 85 48 69 76 62
 27 62 47 18 73 72 19 64 26 10
 55 11 23 89 91 31 22 85 69 11
 99 89 10 52 93 73 31 58 31 10
 98 34 86 93 93 18 40 36 41 23
 62 25 97 12 13 81 52 93 96 29
 57 35 33 60 48 76 62 56 63 68
 34 44 52 34 52 23 99 31 91 21

$ _
```
To build for `ROCm`, you have to specify the compiler:
```shell
CXX=hipcc cmake -S . -B build
cmake --build build -j
```
And that's also true for `oneAPI`:
```shell
CXX=icpx cmake -S . -B build
cmake --build build -j
```
You may need to select target device for `oneAPI` version by setting `ONEAPI_DEVICE_SELECTOR` or `SYCL_DEVICE_FILTER` environment variable first:
```shell
# oneAPI 2023.1.0 or higher
ONEAPI_DEVICE_SELECTOR=[level_zero|opencl|cuda|hip|esimd_emulator|*][:cpu|gpu|fpga|*]

# older versions of oneAPI
SYCL_DEVICE_FILTER=[level_zero|opencl|cuda|hip|esimd_emulator|*][:cpu|gpu|acc|*]
```
You can find the complete syntax [here](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector). For example:
```
$ ONEAPI_DEVICE_SELECTOR=:gpu build/rand100_oneapi

 82 82 42 72 75 37 38 55 88 48
 68 80 78 32 21 13 52 18 53 55
 73 58 80 52 78 85 48 69 76 62
 27 62 47 18 73 72 19 64 26 10
 55 11 23 89 91 31 22 85 69 11
 99 89 10 52 93 73 31 58 31 10
 98 34 86 93 93 18 40 36 41 23
 62 25 97 12 13 81 52 93 96 29
 57 35 33 60 48 76 62 56 63 68
 34 44 52 34 52 23 99 31 91 21

$ _
```
## OpenMP (`rand100_openmp.cpp`) 
```c++
#include <iostream>
#include <iomanip>
#include <vector>

#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long pi_seed{3141592654};
    const auto n{100};
    std::vector<int> v(n);

    p2rng::generate_n
    (   std::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(pi_seed))
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
```
## CUDA/ROCm (`rand100.cu`)
```c++
#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long pi_seed{3141592654};
    const auto n{100};
    thrust::device_vector<int> v(n);

    p2rng::generate_n
    (   std::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(pi_seed)) 
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
```
## oneAPI (`rand100_oneapi.cpp`)
```c++
#include <iostream>
#include <iomanip>

#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>
#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long pi_seed{3141592654};
    const auto n{100};
    sycl::buffer<int> v{sycl::range(n)};
    sycl::queue q;

    p2rng::generate_n
    (   dpl::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(pi_seed))
    ,   q   // this is optional and can be omitted
    ).wait();

    sycl::host_accessor va{v, sycl::read_only};
    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << va[i];
    }
    std::cout << '\n' << std::endl;
}
```
