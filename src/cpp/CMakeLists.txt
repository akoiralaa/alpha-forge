cmake_minimum_required(VERSION 3.18)
project(onebrain_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Static library (used by C++ tests and pybind11 module) ──
add_library(onebrain_core STATIC
    src/feature_engine.cpp
)
target_include_directories(onebrain_core PUBLIC include)

# ── pybind11 Python module ──────────────────────────────────
find_package(pybind11 REQUIRED)

pybind11_add_module(_onebrain_cpp src/bindings.cpp)
target_link_libraries(_onebrain_cpp PRIVATE onebrain_core)

# ── C++ unit tests ──────────────────────────────────────────
option(BUILD_TESTS "Build C++ tests" ON)
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_circular_buffer tests/test_circular_buffer.cpp)
    target_link_libraries(test_circular_buffer PRIVATE onebrain_core)
    add_test(NAME test_circular_buffer COMMAND test_circular_buffer)

    add_executable(test_welford tests/test_welford.cpp)
    target_link_libraries(test_welford PRIVATE onebrain_core)
    add_test(NAME test_welford COMMAND test_welford)

    add_executable(test_feature_engine tests/test_feature_engine.cpp)
    target_link_libraries(test_feature_engine PRIVATE onebrain_core)
    add_test(NAME test_feature_engine COMMAND test_feature_engine)
endif()
