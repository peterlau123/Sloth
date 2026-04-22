cmake_minimum_required(VERSION 3.20...3.26)

project(
  Sloth
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds not allowed. Please create a build directory.")
endif()

option(SLOTH_BUILD_TESTS "Build tests" ON)
option(SLOTH_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SLOTH_ENABLE_TRITON "Enable Triton support" OFF)

include(cmake/CPM.cmake)

CPMAddPackage(
  NAME pybind11
  VERSION 2.11.1
  GIT_REPOSITORY https://github.com/pybind/pybind11
  OPTIONS "PYBIND11_FINDPYTHON OFF"
)

CPMAddPackage(
  NAME cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass
  GIT_TAG v3.5.0
  OPTIONS "CUTLASS_ENABLE_TESTS OFF" "CUTLASS_ENABLE_EXAMPLES OFF"
)

find_package(CUDAToolkit REQUIRED)

set(SLOTH_SOURCES
  common/cpp/utils/timer.cpp
  common/cpp/utils/checker.cpp
)

set(SLOTH_CUDA_SOURCES
  operators/01_elementwise/add/cute/add_kernel.cu
  operators/01_elementwise/add/cutlass/add_kernel.cu
)

add_library(sloth SHARED ${SLOTH_SOURCES} ${SLOTH_CUDA_SOURCES})

target_include_directories(sloth
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/cpp/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(sloth
  PUBLIC
    CUDA::cudart
    cutlass::cutlass
)

target_compile_options(sloth PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>
)

pybind11_add_module(sloth_bindings common/cpp/bindings.cpp)
target_link_libraries(sloth_bindings PRIVATE sloth)

file(GLOB_RECURSE SLOTH_PYTHON_SOURCES "common/python/sloth/*.py")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/operators DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/common/python/sloth/)

install(TARGETS sloth sloth_bindings
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY common/cpp/include/ DESTINATION include)

if(SLOTH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SLOTH_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()