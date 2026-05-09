cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_ARCHITECTURES 75 80)

project(LeetGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
list(APPEND CMAKE_PREFIX_PATH "${THIRD_PARTY_DIR}/gtest")
find_package(GTest REQUIRED)

enable_testing()
file(GLOB_RECURSE ALL_TEST_FILES CONFIGURE_DEPENDS "test/test_*.cu")

add_executable(LeetGPU ${ALL_TEST_FILES})
target_include_directories(LeetGPU PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(LeetGPU PRIVATE GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(LeetGPU)