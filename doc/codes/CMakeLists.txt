cmake_minimum_required(VERSION 3.21...3.26)

include(cmake/BuildType.cmake)

project(rand10-100 CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(TBB REQUIRED)

include(FetchContent)
find_package(p2rng CONFIG
  COMPONENTS openmp
)
if(NOT p2rng_FOUND)
  message(STATUS "Fetching p2rng library...")
  FetchContent_Declare(
    p2rng
    GIT_REPOSITORY https://github.com/arminms/p2rng.git
    GIT_TAG main
  )
  # setting required p2rng components
  set(P2RNG_COMPONENTS openmp
    CACHE STRING "Required components"
  )
  FetchContent_MakeAvailable(p2rng)
endif()

set(PROGS
  rand_10-100_v1
  rand_10-100_v2
  rand_10-100_v3
  rand_10-100_v4
  rand_10-100_v5
  rand_10-100_v6
  rand_10-100_v7
  rand_10-100_v8
)

foreach(PROG ${PROGS})
  add_executable(${PROG} ${PROG}.cpp)
  # target_compile_features(${PROG} PRIVATE cxx_std_17)
  target_link_libraries(${PROG} PRIVATE TBB::tbb p2rng::openmp)
endforeach()
