cmake_minimum_required(VERSION 3.21...3.26)

include(cmake/BuildType.cmake)

## life is about choices...
#
option(P2RNG_ENABLE_TESTS "Enable the unit tests ?" ON)
option(P2RNG_ENABLE_BENCHMARKS "Enable benchmarks ?" ON)

## finally our project...
#
project(
  p2rng
  VERSION 0.1.0
  DESCRIPTION "A modern header-only C++ library for parallel algorithmic (pseudo) random number generation supporting OpenMP, CUDA, ROCm and oneAPI"
  LANGUAGES CXX
)

## prevent in-source builds
#
if (CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "Building in-source is not supported! Create a build dir and remove ${CMAKE_SOURCE_DIR}/CMakeCache.txt")
endif()

## necessary modules
#
include(FetchContent)

## for systems with main libraries in non-standard locations
#
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

## make cache variables for install destinations
#
include(GNUInstallDirs)

## setting target API
#
set(P2RNG_TARGET_API "openmp" CACHE
  STRING "Choose target API ?")
set_property(CACHE P2RNG_TARGET_API PROPERTY STRINGS
  "cuda"
  "oneapi"
  "openmp"
  "rocm"
)

## switch to the target API
#
if(${P2RNG_TARGET_API} STREQUAL cuda)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 11 REQUIRED)
  else()
    message(FATAL_ERROR "No CUDA support found")
  endif()
elseif(${P2RNG_TARGET_API} STREQUAL oneapi)
  if("${CMAKE_CXX_COMPILER}" MATCHES ".*clang\\+\\+")
    set(SYCL_TARGETS "nvptx64-nvidia-cuda,amdgcn-amd-amdhsa,spir64" CACHE STRING "Choose SYCL targets?" FORCE)
    set(SYCL_CUDA_ARCH "sm_50" CACHE STRING "Choose CUDA compute capability for SYCL?" FORCE)
    set(SYCL_ROCM_ARCH "gfx90a" CACHE STRING "Choose SYCL target backend for AMD GPUs?" FORCE)
  else()
    set(SYCL_TARGETS "spir64" CACHE STRING "Choose SYCL targets?" FORCE)
  endif()
  find_package(IntelSYCL QUIET)
  if(NOT IntelSYCL_FOUND)
    find_package(IntelDPCPP QUIET REQUIRED)
  endif()
  find_package(oneDPL REQUIRED)
elseif(${P2RNG_TARGET_API} STREQUAL rocm)
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
  else()
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  endif()
  find_package(hip REQUIRED)
  find_package(rocThrust REQUIRED)
  set(HIP_TARGET "hip::device" CACHE
      STRING "Choose HIP target ?"
  )
  set_property(CACHE HIP_TARGET PROPERTY STRINGS
    "hip::host"
    "hip::device"
  )
elseif(${P2RNG_TARGET_API} STREQUAL openmp)
  find_package(OpenMP REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
else()
  message(FATAL_ERROR "Wrong P2RNG_TARGET_API: ${P2RNG_TARGET_API}")
endif()

## defining p2rng target
#
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

## installing p2rng header files
#
add_subdirectory(include)

## defining p2rng::cuda target
#
add_library(cuda INTERFACE)
add_library(${PROJECT_NAME}::cuda ALIAS cuda)
target_compile_options(cuda INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
)
target_link_libraries(cuda INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
)

## defining p2rng::oneapi target
#
add_library(oneapi INTERFACE)
add_library(${PROJECT_NAME}::oneapi ALIAS oneapi)
target_compile_options(oneapi INTERFACE
  -fsycl
  -fsycl-targets=${SYCL_TARGETS}
  $<$<BOOL:${SYCL_CUDA_ARCH}>:-Xsycl-target-backend=nvptx64-nvidia-cuda;--offload-arch=${SYCL_CUDA_ARCH}>
  $<$<BOOL:${SYCL_ROCM_ARCH}>:-Xsycl-target-backend=amdgcn-amd-amdhsa;--offload-arch=${SYCL_ROCM_ARCH}>
)
target_link_options(oneapi INTERFACE
  -fsycl
  -fsycl-targets=${SYCL_TARGETS}
  $<$<BOOL:${SYCL_CUDA_ARCH}>:-Xsycl-target-backend=nvptx64-nvidia-cuda;--offload-arch=${SYCL_CUDA_ARCH}>
  $<$<BOOL:${SYCL_ROCM_ARCH}>:-Xsycl-target-backend=amdgcn-amd-amdhsa;--offload-arch=${SYCL_ROCM_ARCH}>
)
target_link_libraries(oneapi INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
  oneDPL
)

## defining p2rng::rocm target
#
add_library(rocm INTERFACE)
add_library(${PROJECT_NAME}::rocm ALIAS rocm)
target_compile_options(rocm INTERFACE -Wno-unused-result)
target_link_libraries(rocm INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
  ${HIP_TARGET}
  roc::rocthrust
)

## defining p2rng::openmp target
#
add_library(openmp INTERFACE)
add_library(${PROJECT_NAME}::openmp ALIAS openmp)
target_link_libraries(openmp INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
  OpenMP::OpenMP_CXX
)

## add unit tests
#
if(${P2RNG_ENABLE_TESTS})
  enable_testing()
  find_package(
    Catch2 3
    HINTS $ENV{HOME} /usr/local /opt/local /opt
  )
  if(NOT Catch2_FOUND)
      message(STATUS "Fetching Catch2 library...")
      FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0
      )
      FetchContent_MakeAvailable(Catch2)
      list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif()
  include(Catch)
  add_subdirectory(test)
endif()

## add benchmarks
#
if(${P2RNG_ENABLE_BENCHMARKS})
  find_package(benchmark
    CONFIG
    HINTS $ENV{HOME} /usr/local /opt/local /opt
  )
  if(NOT benchmark_FOUND)
    message(STATUS "Fetching Google Benchmark library...")
    FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.7.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
      "Enable testing of the benchmark library."
    )
    FetchContent_MakeAvailable(benchmark)
  endif()
  add_subdirectory(perf)
endif()
