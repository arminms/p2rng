## defining target for benchmarks
#
add_executable(benchmarks
  benchmarks_${P2RNG_TARGET_API}.$<IF:$<STREQUAL:${P2RNG_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for benchmarks
#
target_link_libraries(benchmarks PRIVATE
  ${PROJECT_NAME}::${P2RNG_TARGET_API}
  benchmark::benchmark
)
