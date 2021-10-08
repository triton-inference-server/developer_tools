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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>

namespace triton {
namespace backend {
namespace rapids {
struct triton_memory_resource final : public rmm::mr::device_memory_resource {
  triton_memory_resource(
    TRITONBACKEND_MemoryManager* manager,
    device_id_t device_id,
    rmm::mr::device_memory_resource* fallback) : manager_{manager}, device_id_{device_id}, fallback_{fallback} {}

  bool supports_streams() const noexcept override { return false; }
  bool supports_get_mem_info() const noexcept override { return false; }
  auto* get_triton_manager() const noexcept { return manager_; }

 private:
  TRITONBACKEND_MemoryManager* manager_;
  std::int64_t device_id_;
  rmm::mr::device_memory_resource* fallback_;

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    auto* ptr = static_cast<void*>(nullptr);
    if (manager_ == nullptr) {
      ptr = fallback_->allocate(bytes, stream);
    } else {
      triton_check(TRITONBACKEND_MemoryManagerAllocate(
        manager_,
        &ptr,
        TRITONSERVER_MEMORY_GPU,
        device_id_,
        static_cast<std::uint64_t>(bytes)
      ));
    }
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) {
    if (manager_ == nullptr) {
      fallback_->deallocate(ptr, bytes, stream);
    } else {
      triton_check(TRITONBACKEND_MemoryManagerFree(
        manager_,
        ptr,
        TRITONSERVER_MEMORY_GPU,
        device_id_
      ));
    }
  }

  bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override
  {
    auto* other_triton_mr = dynamic_cast<triton_memory_resource const*>(&other);
    return (other_triton_mr != nullptr && other_triton_mr->get_triton_manager() == manager_);
  }

  std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view stream) const override {
    throw std::runtime_error("Mem info API not supported by triton_memory_resource");
  }

};

}}}
