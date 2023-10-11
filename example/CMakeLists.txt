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
