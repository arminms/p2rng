#
# Copyright (c) 2023 Armin Sobhani (https://arminsobhani.ca)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
cmake_minimum_required(VERSION 3.21...3.26)

## detect if p2rng is being bundled, disable installations in that case
#
if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else()
  set(NOT_SUBPROJECT OFF)
endif()

## Set default build type to Release if none was specified
#
include(cmake/BuildType.cmake)

## life is about choices...
#
option(P2RNG_ENABLE_TESTS "Enable the unit tests ?" ON)
option(P2RNG_ENABLE_BENCHMARKS "Enable benchmarks ?" ON)

## finally our project...
#
project(
  p2rng
  VERSION 0.6.0
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

## set target API
#
set(P2RNG_TARGET_API "openmp" CACHE
  STRING "Choose target API ?")
set_property(CACHE P2RNG_TARGET_API PROPERTY STRINGS
  "cuda"
  "oneapi"
  "openmp"
  "rocm"
)

## set HIP target
#
set(HIP_TARGET "hip::device" CACHE
  STRING "Choose HIP target ?"
)
set_property(CACHE HIP_TARGET PROPERTY STRINGS
  "hip::host"
  "hip::device"
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
  find_package(IntelDPCPP REQUIRED)
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
elseif(${P2RNG_TARGET_API} STREQUAL openmp)
  find_package(OpenMP REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
else()
  message(FATAL_ERROR "Wrong P2RNG_TARGET_API: ${P2RNG_TARGET_API}")
endif()

## define p2rng target
#
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${p2rng_VERSION}
  SOVERSION ${p2rng_VERSION_MAJOR}
)

## define p2rng::cuda target
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

## define p2rng::oneapi target
#
add_library(oneapi INTERFACE)
add_library(${PROJECT_NAME}::oneapi ALIAS oneapi)
target_link_libraries(oneapi INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
  oneDPL
)

## define p2rng::rocm target
#
add_library(rocm INTERFACE)
add_library(${PROJECT_NAME}::rocm ALIAS rocm)
target_compile_options(rocm INTERFACE -Wno-unused-result)
target_link_libraries(rocm INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
  ${HIP_TARGET}
  roc::rocthrust
)

## define p2rng::openmp target
#
add_library(openmp INTERFACE)
add_library(${PROJECT_NAME}::openmp ALIAS openmp)
target_link_libraries(openmp INTERFACE
  ${PROJECT_NAME}::${PROJECT_NAME}
  OpenMP::OpenMP_CXX
)

## only perform the installation when p2rng is not being used as a subproject
#
if (NOT_SUBPROJECT)
  ## install the target and create export-set
  #
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
  )

  ## install p2rng header files
  #
  add_subdirectory(include)

  ## generate and install export file
  #
  install(
    EXPORT ${PROJECT_NAME}-targets
    FILE ${PROJECT_NAME}-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  ## install the target and create export-set for cuda
  #
  install(
    TARGETS cuda
    EXPORT cuda-targets
  )

  ## generate and install export file for cuda
  #
  install(
    EXPORT cuda-targets
    FILE ${PROJECT_NAME}-cuda-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  ## install the target and create export-set for oneapi
  #
  install(
    TARGETS oneapi
    EXPORT oneapi-targets
  )

  ## generate and install export file for oneapi
  #
  install(
    EXPORT oneapi-targets
    FILE ${PROJECT_NAME}-oneapi-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  ## install the target and create export-set for rocm
  #
  install(
    TARGETS rocm
    EXPORT rocm-targets
  )

  ## generate and install export file for rocm
  #
  install(
    EXPORT rocm-targets
    FILE ${PROJECT_NAME}-rocm-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  ## install the target and create export-set for openmp
  #
  install(
    TARGETS openmp
    EXPORT openmp-targets
  )

  ## generate and install export file for openmp
  #
  install(
    EXPORT openmp-targets
    FILE ${PROJECT_NAME}-openmp-targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  ## generate the config file that includes the exports
  #
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  ## create version file
  #
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    VERSION "${p2rng_VERSION_MAJOR}.${p2rng_VERSION_MINOR}"
    COMPATIBILITY AnyNewerVersion
  )

  ## install p2rng export config files
  #
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

  ## make export target available from the build tree
  #
  export(EXPORT ${PROJECT_NAME}-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-targets.cmake"
    NAMESPACE ${PROJECT_NAME}::
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
endif(NOT_SUBPROJECT)