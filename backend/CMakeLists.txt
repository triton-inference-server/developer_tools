#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.02/RAPIDS.cmake
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
option(BUILD_TESTS "Build developer_tools_backend unit-tests" ON)
option(BUILD_EXAMPLE "Build developer_tools_backend example backend" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable depreaction warnings " ON)
option(NVTX "Enable nvtx markers" OFF)
option(TRITON_ENABLE_STATS "Enable statistics collection in Triton" ON)
set(TRITON_COMMON_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/backend repo")

message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Build DEVELOPER_TOOLS_BACKEND unit-tests: ${BUILD_TESTS}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Enable nvtx markers: ${NVTX}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Enable GPU support: ${TRITON_ENABLE_GPU}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Enable statistics collection in Triton: ${TRITON_ENABLE_STATS}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Triton common repo tag: ${TRITON_COMMON_REPO_TAG}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Triton core repo tag: ${TRITON_CORE_REPO_TAG}")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: Triton backend repo tag: ${TRITON_BACKEND_REPO_TAG}")

##############################################################################
# - Project Initialization ---------------------------------------------------

if(TRITON_ENABLE_GPU)
  rapids_cuda_init_architectures(DEVELOPER_TOOLS_BACKEND)
  project(DEVELOPER_TOOLS_BACKEND VERSION 22.02.00 LANGUAGES CXX CUDA)
else()
  project(DEVELOPER_TOOLS_BACKEND VERSION 22.02.00 LANGUAGES CXX)
endif()


##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(VERBOSE "DEVELOPER_TOOLS_BACKEND: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

##############################################################################
# - Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
      message(STATUS "DEVELOPER_TOOLS_BACKEND: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

##############################################################################
# - compiler options ---------------------------------------------------------
if(TRITON_ENABLE_GPU)
  # * find CUDAToolkit package
  # * determine GPU architectures
  # * enable the CMake CUDA language
  # * set other CUDA compilation flags
  rapids_find_package(CUDAToolkit REQUIRED
      BUILD_EXPORT_SET developer_tools_backend-exports
      INSTALL_EXPORT_SET developer_tools_backend-exports
      )
  include(cmake/modules/ConfigureCUDA.cmake)
endif()

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

add_library(developer_tools_backend INTERFACE)
add_library(developer_tools_backend::developer_tools_backend ALIAS developer_tools_backend)
target_include_directories(developer_tools_backend INTERFACE "$<BUILD_INTERFACE:${DEVELOPER_TOOLS_BACKEND_SOURCE_DIR}/include>"
                                          "$<INSTALL_INTERFACE:include>")

target_link_libraries(developer_tools_backend
INTERFACE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  $<$<BOOL:${TRITON_ENABLE_GPU}>:raft::raft>
  triton-core-serverstub
  triton-backend-utils
)

if (TRITON_ENABLE_GPU)
  target_compile_features(
    developer_tools_backend INTERFACE cxx_std_17
    $<BUILD_INTERFACE:cuda_std_17>
  )
else()
  target_compile_features(
    developer_tools_backend INTERFACE cxx_std_17
  )
endif()

rapids_cmake_install_lib_dir(lib_dir)
install(TARGETS developer_tools_backend
        DESTINATION ${lib_dir}
        EXPORT developer_tools_backend-exports
        )

include(GNUInstallDirs)
install(DIRECTORY include/triton/developer_tools/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/developer_tools
        )

# Temporary install of backend_tools.hpp while the file is removed
install(FILES include/backend_tools.hpp
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/developer_tools
	)

##############################################################################
# - install export -----------------------------------------------------------
set(doc_string
[=[
Provide targets for DEVELOPER_TOOLS_BACKEND.

DEVELOPER_TOOLS_BACKEND is a header-only library designed to make it easier and faster
to implement Triton backends.

]=])

 rapids_export(INSTALL developer_tools_backend
    EXPORT_SET developer_tools_backend-exports
    GLOBAL_TARGETS developer_tools_backend # since we can't hook into EXPORT SETS
    NAMESPACE developer_tools_backend::
    DOCUMENTATION doc_string
    )

##############################################################################
# - build export -------------------------------------------------------------

rapids_export(BUILD developer_tools_backend
    EXPORT_SET developer_tools_backend-exports
    GLOBAL_TARGETS developer_tools_backend # since we can't hook into EXPORT SETS
    LANGUAGES CUDA
    DOCUMENTATION doc_string
    NAMESPACE developer_tools_backend::
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
