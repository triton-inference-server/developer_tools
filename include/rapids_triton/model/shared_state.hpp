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
#include <cuda_runtime_api.h>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <triton/common/triton_json.h>
#include <rapids_triton/batch/batch.hpp>
#include <rapids_triton/tensor/tensor.hpp>
#include <rapids_triton/triton/config.hpp>
#include <rapids_triton/triton/deployment.hpp>
#include <rapids_triton/utils/narrow.hpp>

namespace triton { namespace backend { namespace rapids {
  /**
   * @brief Stores shared state for multiple instances of the same model
   */
  struct SharedModelState {

    virtual void load() {}
    virtual void unload() {}

    explicit SharedModelState(
      common::TritonJSON::Value config) : config_{config},
             max_batch_size_(get_max_batch_size(config)) {}

    template <typename T>
    auto get_config_param(std::string const& name) {
      return get_config_param<T>(name, std::optional<T>{});
    }

    template <typename T>
    auto get_config_param(std::string const& name, T default_value) {
      return get_config_param<T>(name, std::make_optional(default_value));
    }

    private:
      common::TritonJSON::Value config_;
      Batch::size_type max_batch_size_;

      template <typename T>
      auto get_config_param(std::string const& name, std::optional<T> const& default_value) {
        auto result = T{};
        if (name == std::string("max_batch_size")) {
          result = max_batch_size_;
          return result;
        }
        auto json_value = common::TritonJson::Value{};
        if (config_.Find(name.c_str(), &json_value) {
          auto string_repr = std::string{};
          triton_check(json_value.MemberAsString("string_value", &string_repr));

          auto input_stream = std::istringstream{string_repr};

          if (std::is_same<T, bool>::value) {
            input_stream >> std::boolalpha >> result;
          } else {
            input_stream >> result;
          }

          if (input_stream.fail()) {
            if (default_value) {
              result = *default_value;
            } else {
              throw TritonException(
                Error::InvalidArg,
                std::string("Bad input for parameter ") + name
              );
            }
          }
        } else {
          if (default_value) {
            result = *default_value;
          } else {
            throw TritonException(
              Error::InvalidArg,
              std::string("Required parameter ") + name + std::string(" not found in config")
            );
          }
        }

        return result;
      }
  };
}}}  // namespace triton::backend::rapids

