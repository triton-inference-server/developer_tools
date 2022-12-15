#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

##############################################################################
# - Target names -------------------------------------------------------------
set(BACKEND_NAME "rapids_linear")
set(BACKEND_TARGET "triton_${BACKEND_NAME}")


##############################################################################
# - Prepare rapids-cmake -----------------------------------------------------
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.12/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(TRITON_LINEAR_EXAMPLE)

project(TRITON_LINEAR_EXAMPLE VERSION 21.10.00 LANGUAGES CXX CUDA)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_BACKEND_TESTS "Build rapids_triton_linear unit-tests" ON)
option(CUDA_ENABLE_KERNEL_INFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINE_INFO "Enable lineinfo in nvcc" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable depreaction warnings " ON)
option(NVTX "Enable nvtx markers" OFF)

message(VERBOSE "TRITON_LINEAR_EXAMPLE: Enabling detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "TRITON_LINEAR_EXAMPLE: Enabling kernelinfo in nvcc: ${CUDA_ENABLE_KERNEL_INFO}")
message(VERBOSE "TRITON_LINEAR_EXAMPLE: Enabling lineinfo in nvcc: ${CUDA_ENABLE_LINE_INFO}")
message(VERBOSE "TRITON_LINEAR_EXAMPLE: Enabling nvtx markers: ${NVTX}")
message(VERBOSE "TRITON_LINEAR_EXAMPLE: Build TRITON_LINEAR_EXAMPLE unit-tests: ${BUILD_TESTS}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "TRITON_LINEAR_EXAMPLE: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "CUML: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

##############################################################################
# - compiler options ---------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET triton-linear-example-exports
  INSTALL_EXPORT_SET triton-linear-example-exports
)
include(cmake/modules/ConfigureCUDA.cmake)

##############################################################################
# - Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

# TODO(wphicks)
include(cmake/thirdparty/get_rapids-triton.cmake)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()


##############################################################################
# - install targets-----------------------------------------------------------

add_library(
  ${BACKEND_TARGET} SHARED
  src/api.cc
  src/gpu_infer.cu
)

set_target_properties(${BACKEND_TARGET}
PROPERTIES BUILD_RPATH                         "\$ORIGIN"
           # set target compile options
           CXX_STANDARD                        17
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       17
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(${BACKEND_TARGET}
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_BACKEND_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_BACKEND_CUDA_FLAGS}>"
)

target_include_directories(${BACKEND_TARGET}
  PRIVATE  "$<BUILD_INTERFACE:${RAPIDS_TRITON_BACKEND_SOURCE_DIR}/include>"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(${BACKEND_TARGET}
PRIVATE
  rapids_triton::rapids_triton
  triton-core-serverstub
  triton-backend-utils
  "${TRITONSERVER_LIB}"
  $<TARGET_NAME_IF_EXISTS:conda_env>
)

install(
  TARGETS ${BACKEND_TARGET}
  LIBRARY DESTINATION /opt/tritonserver/backends/${BACKEND_NAME}
)

##############################################################################
# - build test executable ----------------------------------------------------

# TODO (wphicks)
# if(BUILD_TESTS)
#   include(test/CMakeLists.txt)
# endif()
