# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)

project(tritoncapiwrapper LANGUAGES C CXX)

#
# Options
#
# [FIXME] add option needed for wrapper
option(TRITON_BUILD_TEST "Include unit test for the wrapper" ON)
# [FIXME] doesn't have example yet
# option(TRITON_ENABLE_EXAMPLES "Include examples in build" ON)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)

set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-core)

#
# Triton Server C-API Wrapper
#
file(GLOB SRC_FILES src/*.cc)

add_library(
  triton-server-wrapper
  ${SRC_FILES}
)

add_library(
  TritonServerWrapper::triton-server-wrapper ALIAS triton-server-wrapper
)

target_include_directories(
  triton-server-wrapper
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
      "If the target system is not Windows 10, please update _WIN32_WINNT "
      "to corresponding value.")
endif()
target_compile_features(triton-server-wrapper PRIVATE cxx_std_11)
target_compile_options(
  triton-server-wrapper
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

set_target_properties(
  triton-server-wrapper PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME tritonserverwrapper
)

target_link_libraries(
  triton-server-wrapper
  PUBLIC
    triton-core-serverapi         # from repo-core
    triton-core-serverstub        # from repo-core
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonServerWrapper)

install(
  TARGETS
    triton-server-wrapper
  EXPORT
    triton-server-wrapper-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT
    triton-server-wrapper-targets
  FILE
    TritonServerWrapperTargets.cmake
  NAMESPACE
    TritonServerWrapper::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonServerWrapperConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonServerWrapperConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonServerWrapperConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-server-wrapper-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonServerWrapperTargets.cmake
  NAMESPACE TritonServerWrapper::
)

export(PACKAGE TritonServerWrapper)

if(${TRITON_BUILD_TEST})
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
  )
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(test test)
endif()

if(${TRITON_ENABLE_EXAMPLES})
  add_subdirectory(examples)
endif() # TRITON_ENABLE_EXAMPLES
