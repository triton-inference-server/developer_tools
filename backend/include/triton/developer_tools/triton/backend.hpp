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
#include <cstdint>
#include <triton/developer_tools/exceptions.hpp>
#include <triton/developer_tools/triton/logging.hpp>
#include <string>

namespace triton {
namespace developer_tools {
namespace backend {
inline auto get_backend_name(TRITONBACKEND_Backend& backend)
{
  const char* cname;
  triton_check(TRITONBACKEND_BackendName(&backend, &cname));
  return std::string(cname);
}

namespace {
struct backend_version {
  std::uint32_t major;
  std::uint32_t minor;
};
}  // namespace

inline auto check_backend_version(TRITONBACKEND_Backend& backend)
{
  auto version = backend_version{};
  triton_check(TRITONBACKEND_ApiVersion(&version.major, &version.minor));

  log_info(__FILE__, __LINE__) << "Triton TRITONBACKEND API version: " << version.major << "."
                               << version.minor;

  auto name = get_backend_name(backend);

  log_info(__FILE__, __LINE__) << "'" << name
                               << "' TRITONBACKEND API version: " << TRITONBACKEND_API_VERSION_MAJOR
                               << "." << TRITONBACKEND_API_VERSION_MINOR;

  return ((version.major == TRITONBACKEND_API_VERSION_MAJOR) &&
          (version.minor >= TRITONBACKEND_API_VERSION_MINOR));
}
}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
