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
  triton_rapids-identity SHARED
  src/api.cc
)

if(TRITON_ENABLE_GPU)
  set_target_properties(triton_rapids-identity
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
  set_target_properties(triton_rapids-identity
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             # set target compile options
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             POSITION_INDEPENDENT_CODE           ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endif()

target_compile_options(triton_rapids-identity
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDS_TRITON_CXX_FLAGS}>"
                "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDS_TRITON_CUDA_FLAGS}>"
)

target_include_directories(triton_rapids-identity
  PRIVATE  "$<BUILD_INTERFACE:${RAPIDS_TRITON_SOURCE_DIR}/include>"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(triton_rapids-identity
PRIVATE
  $<$<BOOL:${TRITON_ENABLE_GPU}>:rmm::rmm>
  $<$<BOOL:${TRITON_ENABLE_GPU}>:raft::raft>
  triton-core-serverstub
  triton-backend-utils
  "${TRITONSERVER_LIB}"
  $<TARGET_NAME_IF_EXISTS:conda_env>
)

install(
  TARGETS triton_rapids-identity
  LIBRARY DESTINATION /opt/tritonserver/backends/rapids-identity
)
