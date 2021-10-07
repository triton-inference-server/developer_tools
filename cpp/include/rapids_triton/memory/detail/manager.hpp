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
#include <memory>
#include <mutex>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace triton {
namespace backend {
namespace rapids {
namespace detail {

  inline auto& manager_lock() {
    static auto lock = std::mutex{};
    return lock;
  }

  struct manager_data {
    manager_data() : base_mr_{},
                     pool_mr_{&base_mr_} {}
    auto* get_resource() {
      return &pool_mr_;
    }
   private:
    rmm::mr::cuda_memory_resource base_mr_;
    rmm::mr::pool_memory_resource pool_mr_;
  };

  auto& all_device_managers() {
    // This vector keeps the underlying memory resource objects in-scope until
    // the backend is unloaded. This ensures that they will not be destroyed
    // while a model is still making use of them.
    static auto device_managers = std::vector<manager_data>{};
    return device_managers;
  }

  auto is_default_resource (rmm::cuda_device_id const& device_id) {
    return rmm::get_per_device_resource(device_id)->is_equal(rmm::cuda_memory_resource{});
  }

  auto setup_memory_resource(rmm::cuda_device_id device_id) {
    auto lock = std::lock_guard<std::mutex>{manager_lock()};
    if (is_default_resource(device_id)) {
      auto& device_managers = all_device_managers();
      device_managers.push_back(manager_data{});
      rmm::mr::set_per_device_resource(
          rmm::cuda_device_id{device_id}, device_managers.back()->get_resource());
    }

    return rmm::get_per_device_resource(rmm_device_id);
  }
}  // namespace detail

auto* get_memory_resource(device_id_t device_id) {
  auto result = static_cast<rmm::device_memory_resource*>(nullptr);
  auto rmm_device_id = rmm::cuda_device_id{device_id};
  if (is_default_resource(rmm_device_id)) {
    result = setup_memory_resource(rmm_device_id);
  } else {
    result = rmm::get_per_device_resource(rmm_device_id);
  }
  return result;
}

auto* get_memory_resource() {
  auto device_id = int{};
  cuda_check(cudaGetDevice(&device_id));

  return get_memory_resource(device_id);
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
