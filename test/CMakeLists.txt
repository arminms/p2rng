## defining target for unit tests
#
add_executable(unit_tests
  unit_tests_${P2RNG_TARGET_API}.$<IF:$<STREQUAL:${P2RNG_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for unit tests
#
target_link_libraries(unit_tests PRIVATE
  ${PROJECT_NAME}::${P2RNG_TARGET_API}
  Catch2::Catch2WithMain
)

## finally adding unit tests
#
catch_discover_tests(unit_tests)
