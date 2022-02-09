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
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#else
#include <rapids_triton/cpu_only/cuda_runtime_replacement.hpp>
#endif
#include <algorithm>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <triton/backend/backend_common.h>
#include <rapids_triton/batch/batch.hpp>
#include <rapids_triton/tensor/tensor.hpp>
#include <rapids_triton/triton/config.hpp>
#include <rapids_triton/triton/deployment.hpp>
#include <rapids_triton/utils/narrow.hpp>

namespace triton {
namespace backend {
namespace rapids {
/**
 * @brief Stores shared state for multiple instances of the same model
 */
struct SharedModelState {
  virtual void load() {}
  virtual void unload() {}

  explicit SharedModelState(std::unique_ptr<common::TritonJson::Value>&& config,
                            bool squeeze_output = false)
    : config_{std::move(config)},
      max_batch_size_{get_max_batch_size(*config_)},
      output_shapes_([this, squeeze_output]() {
        auto result         = std::vector<std::pair<std::string, std::vector<std::int64_t>>>{};
        auto output_entries = triton::common::TritonJson::Value{};
        triton_check(config_->MemberAsArray("output", &output_entries));

        result.reserve(output_entries.ArraySize());

        // Using a raw loop because TritonJSON::Value access has no iterator interface
        for (std::size_t i = 0; i < output_entries.ArraySize(); ++i) {
          auto output_entry = triton::common::TritonJson::Value{};
          triton_check(output_entries.IndexAsObject(i, &output_entry));
          auto name = std::string{};
          triton_check(output_entry.MemberAsString("name", &name));

          auto shape         = std::vector<std::int64_t>{};
          auto reshape_entry = triton::common::TritonJson::Value{};
          if (output_entry.Find("reshape", &reshape_entry)) {
            ParseShape(reshape_entry, "shape", &shape);
          } else {
            ParseShape(output_entry, "dims", &shape);
          }
          if (shape[0] != -1) { shape.insert(shape.begin(), -1); }
          // The squeeze_output option was introduced to handle a bad choice of
          // convention in the original FIL backend implementation. For legacy
          // compatibility, we introduced this option into RAPIDS-Triton, but
          // in general, new backends are advised to avoid using it and defer
          // this sort of flattening operation to the consumer.
          if (squeeze_output) {
            shape.erase(std::remove(shape.begin(), shape.end(), std::int64_t{1}), shape.end());
          }
          result.insert(
            std::upper_bound(std::begin(output_shapes_),
                             std::end(output_shapes_),
                             name,
                             [](auto& value, auto& entry) { return value < entry.first; }),
            {name, shape});
        }

        return result;
      }())
  {
  }

  template <typename T>
  auto get_config_param(std::string const& name)
  {
    return get_config_param<T>(name, std::optional<T>{});
  }

  template <typename T>
  auto get_config_param(std::string const& name, T default_value)
  {
    return get_config_param<T>(name, std::make_optional(default_value));
  }

  auto get_output_shape(std::string const& name) const
  {
    auto cached_shape = std::lower_bound(
      std::begin(output_shapes_), std::end(output_shapes_), name, [](auto& entry, auto& value) {
        return entry.first < value;
      });
    if (cached_shape == std::end(output_shapes_)) {
      auto log_stream = std::stringstream{};
      log_stream << "No output with name " << name << " in configuration.";
      throw TritonException(Error::Internal, log_stream.str());
    } else {
      return cached_shape->second;
    }
  }

  auto get_output_names() const {
    auto output_names = std::vector<std::string>{output_shapes_.size()};
    std::for_each(std::begin(output_shapes_), std::end(output_shapes_), [&output_names](auto& output_shape) {
      output_names.push_back(output_shape.first);
    });
    return output_names;
  }

  auto check_output_name(std::string const& name) const -> bool {
    auto cached_shape = std::lower_bound(
      std::begin(output_shapes_), std::end(output_shapes_), name, [](auto& entry, auto& value) {
        return entry.first < value;
      });
    return cached_shape != std::end(output_shapes_);
  }

 private:
  std::unique_ptr<common::TritonJson::Value> config_;
  Batch::size_type max_batch_size_;
  std::vector<std::pair<std::string, std::vector<std::int64_t>>> mutable output_shapes_;

  template <typename T>
  auto get_config_param(std::string const& name, std::optional<T> const& default_value)
  {
    auto result = T{};
    if (name == std::string("max_batch_size")) {
      result = max_batch_size_;
      return result;
    }
    auto parameters = common::TritonJson::Value{};
    auto json_value = common::TritonJson::Value{};
    if (config_->Find("parameters", &parameters) && parameters.Find(name.c_str(), &json_value)) {
      auto string_repr = std::string{};
      triton_check(json_value.MemberAsString("string_value", &string_repr));

      auto input_stream = std::istringstream{string_repr};

      if constexpr (std::is_same_v<T, bool>) {
        input_stream >> std::boolalpha >> result;
      } else {
        input_stream >> result;
      }

      if (input_stream.fail()) {
        if (default_value) {
          result = *default_value;
        } else {
          throw TritonException(Error::InvalidArg, std::string("Bad input for parameter ") + name);
        }
      }
    } else {
      if (default_value) {
        result = *default_value;
      } else {
        throw TritonException(
          Error::InvalidArg,
          std::string("Required parameter ") + name + std::string(" not found in config"));
      }
    }

    return result;
  }
};
}  // namespace rapids
}  // namespace backend
}  // namespace triton
