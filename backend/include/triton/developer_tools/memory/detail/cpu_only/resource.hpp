/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <triton/core/tritonbackend.h>

#include <triton/developer_tools/memory/detail/resource.hpp>
#include <triton/developer_tools/triton/device.hpp>

namespace triton {
namespace developer_tools {
namespace backend {
namespace detail {

template<>
inline void setup_memory_resource<false>(device_id_t device_id,
                                         TRITONBACKEND_MemoryManager* triton_manager) { }

}  // namespace detail
}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
