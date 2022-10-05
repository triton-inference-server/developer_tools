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
add_library(
  triton_tools-identity SHARED
  src/api.cc
)

if(TRITON_ENABLE_GPU)
  set_target_properties(triton_tools-identity
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
  set_target_properties(triton_tools-identity
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(triton_tools-identity
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${DEVELOPER_TOOLS_BACKEND_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${DEVELOPER_TOOLS_BACKEND_CUDA_FLAGS}>"
)

target_include_directories(triton_tools-identity
  PRIVATE  "$<BUILD_INTERFACE:${DEVELOPER_TOOLS_BACKEND_SOURCE_DIR}/include>"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(triton_tools-identity
PRIVATE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  $<$<BOOL:${TRITON_ENABLE_GPU}>:raft::raft>
  triton-core-serverstub
  triton-backend-utils
  $<TARGET_NAME_IF_EXISTS:conda_env>
)

install(
  TARGETS triton_tools-identity
  LIBRARY DESTINATION /opt/tritonserver/backends/tools-identity
)
