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

#include <cuda_runtime_api.h>
#include <gpu_infer.h>
#include <names.h>
#include <shared_state.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <triton_backend/batch/batch.hpp>        // dev_tools::Batch
#include <triton_backend/memory/buffer.hpp>      // dev_tools::Buffer, dev_tools::copy
#include <triton_backend/memory/types.hpp>       // dev_tools::MemoryType
#include <triton_backend/model/model.hpp>        // dev_tools::Model
#include <triton_backend/tensor/tensor.hpp>      // dev_tools::copy
#include <triton_backend/triton/deployment.hpp>  // dev_tools::DeploymentType
#include <triton_backend/triton/device.hpp>      // dev_tools::device_id_t
#include <triton_backend/triton/logging.hpp>     // dev_tools::log_info
#include <sstream>

namespace triton {
namespace backend {
namespace NAMESPACE {

struct RapidsModel : dev_tools::Model<RapidsSharedState> {
  RapidsModel(std::shared_ptr<RapidsSharedState> shared_state,
              dev_tools::device_id_t device_id, cudaStream_t default_stream,
              dev_tools::DeploymentType deployment_type,
              std::string const& filepath)
      : dev_tools::Model<RapidsSharedState>(shared_state, device_id,
                                         default_stream, deployment_type,
                                         filepath) {}

  void predict(dev_tools::Batch& batch) const {
    auto u = get_input<float>(batch, "u");
    auto v = get_input<float>(batch, "v");

    auto r = get_output<float>(batch, "r");

    auto alpha = get_shared_state()->alpha;
    if (u.mem_type() == dev_tools::HostMemory) {
      for (auto i = std::size_t{}; i < u.size(); ++i) {
        r.data()[i] =
            alpha * u.data()[i] + v.data()[i] + c.data()[i % c.size()];
      }
    } else {
      gpu_infer(r.data(), u.data(), v.data(), c.data(), alpha, c.size(),
                u.size(), r.stream());
    }

    r.finalize();
  }

  void load() {
    auto path = std::filesystem::path(get_filepath());
    /* If the config file does not specify a filepath for the model,
     * get_filepath returns the directory where the serialized model should be
     * found. It is generally good practice to provide logic to allow the use
     * of a default filename so that model configurations do not always have to
     * specify a path to their model */
    if (std::filesystem::is_directory(path)) {
      path /= "c.txt";
    }

    // Read space-separated text file into a vector of floats
    auto model_vec = std::vector<float>{};
    auto model_file = std::ifstream(path.string());
    auto input_line = std::string{};
    std::getline(model_file, input_line);
    auto input_stream = std::stringstream{input_line};
    auto value = 0.0f;
    while (input_stream >> value) {
      model_vec.push_back(value);
    }

    // Construct buffer to hold c based on details of this model deployment
    auto memory_type = dev_tools::MemoryType{};
    if constexpr (dev_tools::IS_GPU_BUILD) {
      if (get_deployment_type() == dev_tools::GPUDeployment) {
        memory_type = dev_tools::DeviceMemory;
      } else {
        memory_type = dev_tools::HostMemory;
      }
    } else {
      memory_type = dev_tools::HostMemory;
    }

    c = dev_tools::Buffer<float>(model_vec.size(), memory_type, get_device_id(),
                              get_stream());

    /* Use a Buffer view on model_vec to safely copy data to its final
     * location. Making use of dev_tools::copy here provides additional safety
     * checks to avoid buffer overruns. Note that the destination buffer comes
     * first in dev_tools::copy calls, so we are copying *into* c */
    dev_tools::copy(c, dev_tools::Buffer<float>(model_vec.data(), model_vec.size(),
                                          dev_tools::HostMemory));
  }

 private:
  dev_tools::Buffer<float> c{};
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
