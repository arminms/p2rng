# Example
This folder contains a complete example of how to use `p2rng` including the `CMakeLists.txt` file. There are three source files for different APIs that doing the exact same thing:
```shell
└── example
    ├── CMakeLists.txt
    ├── README.md
    ├── rand100.cu
    ├── rand100_oneapi.cpp
    ├── rand100_openmp.cpp
    └── rand100_rocm.cpp
```
## CMake script (`CMakeLists.txt`)
The CMake script builds the `OpenMP` version by default. Depending on the availability of other APIs on the system, it tries to build them as well and ignores otherwise. That's why `cuda`, `oneapi` and `rocm` are listed as `OPTIONAL_COMPONENTS` in `find_package(p2rng)` command:
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
  # setting required p2rng components
  set(P2RNG_COMPONENTS openmp cuda oneapi rocm
    CACHE STRING "Required components"
  )
  FetchContent_MakeAvailable(p2rng)
endif()

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
  add_executable(rand100_rocm rand100_rocm.cpp)
  target_link_libraries(rand100_rocm PRIVATE p2rng::rocm)
endif()

## build oneAPI version if it's available, go with OpenMP otherwise
#
find_package(IntelDPCPP CONFIG)
if (IntelDPCPP_FOUND)
  add_executable(rand100_oneapi rand100_oneapi.cpp)
  target_link_libraries(rand100_oneapi PRIVATE p2rng::oneapi)
else() # openmp and oneapi are mutually exclusive
  add_executable(rand100_openmp rand100_openmp.cpp)
  target_link_libraries(rand100_openmp PRIVATE p2rng::openmp)
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

 10 24 36 72 10 94 48 91 49 94
 75 51 27 40 47 93 20 21 81 41
 14 32 60 42 34 41 41 36 69 78
 90 48 18 42 36 89 49 36 77 13
 34 61 14 88 94 67 67 97 71 40
 82 46 26 61 34 60 81 16 63 91
 43 42 69 65 13 70 94 12 81 71
 84 59 44 76 96 86 39 64 10 89
 85 57 84 89 78 39 16 64 24 11
 99 25 41 21 20 11 93 13 49 54

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

 10 24 36 72 10 94 48 91 49 94
 75 51 27 40 47 93 20 21 81 41
 14 32 60 42 34 41 41 36 69 78
 90 48 18 42 36 89 49 36 77 13
 34 61 14 88 94 67 67 97 71 40
 82 46 26 61 34 60 81 16 63 91
 43 42 69 65 13 70 94 12 81 71
 84 59 44 76 96 86 39 64 10 89
 85 57 84 89 78 39 16 64 24 11
 99 25 41 21 20 11 93 13 49 54

$ _
```
## OpenMP (`rand100_openmp.cpp`) 
```c++
#include <iostream>
#include <iomanip>
#include <vector>

#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long seed{2718281828};
    const auto n{100};
    std::vector<int> v(n);

    p2rng::generate_n
    (   std::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(seed))
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
```
## CUDA (`rand100.cu`)
```c++
#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long seed{2718281828};
    const auto n{100};
    thrust::device_vector<int> v(n);

    p2rng::cuda::generate_n
    (   std::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(seed)) 
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
```
## ROCm (`rand100_rocm.cpp`)
```c++
#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long pi_seed{2718281828};
    const auto n{100};
    thrust::device_vector<int> v(n);

    p2rng::rocm::generate_n
    (   std::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(seed)) 
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
{   const unsigned long seed{2718281828};
    const auto n{100};
    sycl::buffer<int> v{sycl::range(n)};
    sycl::queue q;

    p2rng::oneapi::generate_n
    (   dpl::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(seed))
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
