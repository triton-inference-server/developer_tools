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

#include <gtest/gtest.h>

#include <iostream>
#include <triton/developer_tools/triton/logging.hpp>

namespace triton {
namespace developer_tools {
namespace backend {
TEST(BackendTools, logging)
{
  log_debug("Debug test message");
  log_info("Info test message");
  log_warn("Warn test message");
  log_error("Error test message");
}

TEST(BackendTools, stream_logging)
{
  log_debug() << "Streamed debug test message";
  log_info() << "Streamed info test message";
  log_warn() << "Streamed warn test message";
  log_error() << "Streamed error test message";
}

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
