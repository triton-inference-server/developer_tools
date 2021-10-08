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
#include <deque>
#include <memory>
#include <mutex>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rapids_triton/triton/triton_memory_resource.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <triton/core/tritonbackend.h>

namespace triton {
namespace backend {
namespace rapids {
namespace detail {

  inline auto& resource_lock() {
    static auto lock = std::mutex{};
    return lock;
  }

  /** A struct used solely to keep memory resources in-scope for the lifetime
   * of the backend */
  struct resource_data {
    resource_data() : base_mr_{},
                      triton_mrs_{} {}
    auto* make_new_resource(device_id_t device_id, TRITONBACKEND_MemoryManager* manager) {
      if (manager == nullptr && triton_mrs_.size() != 0) {
        manager = triton_mrs_.back().get_triton_manager();
      }
      triton_mrs_.emplace_back(manager, device_id, &base_mr_);
      return &(triton_mrs_.back());
    }
   private:
    rmm::mr::cuda_memory_resource base_mr_;
    std::deque<triton_memory_resource> triton_mrs_;
  };

  inline auto& get_device_resources() {
    static auto device_resources = resource_data{};
    return device_resources;
  }

  inline auto is_triton_resource (rmm::cuda_device_id const& device_id) {
    auto* triton_mr = dynamic_cast<triton_memory_resource*>(
      rmm::mr::get_per_device_resource(device_id)
    );
    return (triton_mr != nullptr && triton_mr->get_triton_manager() != nullptr);
  }
}  // namespace detail

  inline auto* setup_memory_resource(device_id_t device_id, TRITONBACKEND_MemoryManager* triton_manager=nullptr) {
    auto lock = std::lock_guard<std::mutex>{detail::resource_lock()};
    auto rmm_device_id = rmm::cuda_device_id{device_id};

    if (!detail::is_triton_resource(rmm_device_id)) {
      auto& device_resources = detail::get_device_resources();
      rmm::mr::set_per_device_resource(
        rmm_device_id, device_resources.make_new_resource(device_id, triton_manager));
    }

    return rmm::mr::get_per_device_resource(rmm_device_id);
  }

  inline auto* get_memory_resource(device_id_t device_id) {
    auto rmm_device_id = rmm::cuda_device_id{device_id};
    return rmm::mr::get_per_device_resource(rmm_device_id);
  }

}  // namespace rapids
}  // namespace backend
}  // namespace triton
