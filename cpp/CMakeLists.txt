#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.10/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

##############################################################################
# - User Options  ------------------------------------------------------------

option(TRITON_ENABLE_GPU "Enable GPU support in Triton" ON)
option(BUILD_TESTS "Build rapids_triton unit-tests" ON)
option(BUILD_EXAMPLE "Build rapids_identity example backend" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable depreaction warnings " ON)
option(NVTX "Enable nvtx markers" OFF)
option(TRITON_ENABLE_STATS "Enable statistics collection in Triton" ON)
set(TRITON_COMMON_REPO_TAG "r21.08" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r21.08" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r21.08" CACHE STRING "Tag for triton-inference-server/backend repo")

message(VERBOSE "RAPIDS_TRITON: Build RAPIDS_TRITON unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAPIDS_TRITON: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAPIDS_TRITON: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "RAPIDS_TRITON: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAPIDS_TRITON: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAPIDS_TRITON: Enable nvtx markers: ${NVTX}")
message(VERBOSE "RAPIDS_TRITON: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "RAPIDS_TRITON: Enable GPU support: ${TRITON_ENABLE_GPU}")
message(VERBOSE "RAPIDS_TRITON: Enable statistics collection in Triton: ${TRITON_ENABLE_STATS}")
message(VERBOSE "RAPIDS_TRITON: Triton common repo tag: ${TRITON_COMMON_REPO_TAG}")
message(VERBOSE "RAPIDS_TRITON: Triton core repo tag: ${TRITON_CORE_REPO_TAG}")
message(VERBOSE "RAPIDS_TRITON: Triton backend repo tag: ${TRITON_BACKEND_REPO_TAG}")

##############################################################################
# - Project Initialization ---------------------------------------------------

if(TRITON_ENABLE_GPU)
  rapids_cuda_init_architectures(RAPIDS_TRITON)
endif()

project(RAPIDS_TRITON VERSION 21.10.00 LANGUAGES CXX CUDA)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "RAPIDS_TRITON: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "RAPIDS_TRITON: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

##############################################################################
# - compiler options ---------------------------------------------------------
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET rapids_triton-exports
    INSTALL_EXPORT_SET rapids_triton-exports
    )
include(cmake/modules/ConfigureCUDA.cmake)

##############################################################################
# - Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

if(TRITON_ENABLE_GPU)
  include(cmake/thirdparty/get_rmm.cmake)
  include(cmake/thirdparty/get_raft.cmake)
endif()

include(cmake/thirdparty/get_rapidjson.cmake)
include(cmake/thirdparty/get_triton.cmake)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

##############################################################################
# - install targets-----------------------------------------------------------

add_library(rapids_triton INTERFACE)
add_library(rapids_triton::rapids_triton ALIAS rapids_triton)
target_include_directories(rapids_triton INTERFACE "$<BUILD_INTERFACE:${RAPIDS_TRITON_SOURCE_DIR}/include>"
                                          "$<INSTALL_INTERFACE:include>")

target_link_libraries(rapids_triton
INTERFACE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  $<$<BOOL:${TRITON_ENABLE_GPU}>:raft::raft>
  triton-core-serverstub
  triton-backend-utils
)

target_compile_features(
  rapids_triton INTERFACE cxx_std_17
  $<BUILD_INTERFACE:cuda_std_17>
)

rapids_cmake_install_lib_dir(lib_dir)
install(TARGETS rapids_triton
        DESTINATION ${lib_dir}
        EXPORT rapids_triton-exports
        )

include(GNUInstallDirs)
install(DIRECTORY include/rapids_triton/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rapids_triton
        )

# Temporary install of rapids_triton.hpp while the file is removed
install(FILES include/rapids_triton.hpp
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rapids_triton
	)

##############################################################################
# - install export -----------------------------------------------------------
set(doc_string
[=[
Provide targets for RAPIDS_TRITON.

RAPIDS_TRITON is a header-only library designed to make it easier and faster
to integrate RAPIDS algorithms as Triton backends.

]=])

 rapids_export(INSTALL rapids_triton
    EXPORT_SET rapids_triton-exports
    GLOBAL_TARGETS rapids_triton # since we can't hook into EXPORT SETS
    NAMESPACE rapids_triton::
    DOCUMENTATION doc_string
    )

##############################################################################
# - build export -------------------------------------------------------------

rapids_export(BUILD rapids_triton
    EXPORT_SET rapids_triton-exports
    GLOBAL_TARGETS rapids_triton # since we can't hook into EXPORT SETS
    LANGUAGES CUDA
    DOCUMENTATION doc_string
    NAMESPACE rapids_triton::
    )

##############################################################################
# - build test executable ----------------------------------------------------

if(BUILD_TESTS)
  include(test/CMakeLists.txt)
endif()

##############################################################################
# - build example backend ----------------------------------------------------

if(BUILD_EXAMPLE)
  include(src/CMakeLists.txt)
endif()

##############################################################################
# - doxygen targets ----------------------------------------------------------

# TODO(wphicks)
# include(cmake/doxygen.cmake)
# add_doxygen_target(IN_DOXYFILE Doxyfile.in
#   OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#   CWD ${CMAKE_CURRENT_BINARY_DIR})
