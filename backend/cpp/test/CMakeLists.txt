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
add_executable(test_triton_backend
    test/batch/batch.cpp
    test/build_control.cpp
    test/exceptions.cpp
    test/memory/buffer.cpp
    test/memory/detail/copy.cpp
    test/memory/detail/owned_device_buffer.cpp
    test/memory/resource.cpp
    test/memory/types.cpp
    test/tensor/dtype.cpp
    test/tensor/tensor.cpp
    test/test.cpp
    test/triton/api/execute.cpp
    test/triton/api/initialize.cpp
    test/triton/api/instance_finalize.cpp
    test/triton/api/instance_initialize.cpp
    test/triton/api/model_finalize.cpp
    test/triton/api/model_initialize.cpp
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
    test/triton/statistics.cpp
    test/utils/const_agnostic.cpp
    test/utils/narrow.cpp
)

IF(TRITON_ENABLE_GPU)
  set_target_properties(test_triton_backend
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             CUDA_STANDARD                       17
             CUDA_STANDARD_REQUIRED              ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
else()
  set_target_properties(test_triton_backend
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(test_triton_backend
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${DEV_TOOLS_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${DEV_TOOLS_CUDA_FLAGS}>"
)

target_include_directories(test_triton_backend
    PUBLIC  "$<BUILD_INTERFACE:${DEV_TOOLS_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${DEV_TOOLS_SOURCE_DIR}/test>"
)


find_library(
  TRITONSERVER_LIB
  tritonserver
  PATHS /opt/tritonserver/lib
)

target_link_libraries(test_triton_backend
PRIVATE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  $<$<BOOL:${TRITON_ENABLE_GPU}>:raft::raft>
  triton-core-serverstub
  triton-backend-utils
  gmock
  gmock_main
  GTest::gtest
  GTest::gtest_main
  "${TRITONSERVER_LIB}"
  $<TARGET_NAME_IF_EXISTS:conda_env>
)
