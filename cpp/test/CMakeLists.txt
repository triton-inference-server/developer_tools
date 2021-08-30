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

# keep the files in alphabetical order!
add_executable(test_rapids_triton
    test/build_control.cpp
    test/exceptions.cpp
    test/memory/buffer.cpp
    test/memory/detail/allocate.cpp
    test/memory/detail/copy.cpp
    test/memory/types.cpp
    test/tensor/dtype.cpp
    test/tensor/tensor.cpp
    test/triton/backend.cpp
    test/triton/config.cpp
    test/triton/deployment.cpp
    test/triton/device.cpp
    test/triton/input.cpp
    test/triton/logging.cpp
    test/triton/model.cpp
    test/triton/model_instance.cpp
    test/triton/requests.cpp
    test/triton/responses.cpp
    test/test.cpp
    test/utils/const_agnostic.cpp
    test/utils/narrow.cpp
)

set_target_properties(test_rapids_triton
PROPERTIES BUILD_RPATH                         "\$ORIGIN"
           # set target compile options
           CXX_STANDARD                        17
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       17
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(test_rapids_triton
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_CUDA_FLAGS}>"
)

target_include_directories(test_rapids_triton
    PUBLIC  "$<BUILD_INTERFACE:${RAPIDS_TRITON_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${RAPIDS_TRITON_SOURCE_DIR}/test>"
)


find_library(
  TRITONSERVER_LIB
  tritonserver
  PATHS /opt/tritonserver/lib
)

target_link_libraries(test_rapids_triton
PRIVATE
  rmm::rmm
  raft::raft
  triton-core-serverstub
  triton-backend-utils
  gmock
  gmock_main
  GTest::gtest
  GTest::gtest_main
  "${TRITONSERVER_LIB}"
  $<TARGET_NAME_IF_EXISTS:conda_env>
)
