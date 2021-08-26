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
#include <rapids_triton/exceptions.hpp>
#include <triton/core/tritonserver.h>

#include <string>

namespace triton { namespace backend { namespace rapids {

  namespace {
    /** Log message at indicated level */
    inline void log(
        TRITONSERVER_LogLevel level, const char* filename, const int line,
        const char* message) {
      triton_check(TRITONSERVER_LogMessage(level, filename, line, message));
    }
  }


  /** Log message at INFO level */
  inline void log_info(const char* filename, const int line, const char* message) {
    log(TRITONSERVER_LOG_INFO, filename, line, message);
  }
  inline void log_info(const char* filename, const int line, std::string const& message) {
    log_info(filename, line, message.c_str());
  }
  inline void log_info(const char* message) {
    log_info(__FILE__, __LINE__, message);
  }
  inline void log_info(std::string const& message) {
    log_info(__FILE__, __LINE__, message.c_str());
  }


  /** Log message at WARN level */
  inline void log_warn(const char* filename, const int line, const char* message) {
    log(TRITONSERVER_LOG_WARN, filename, line, message);
  }
  inline void log_warn(const char* filename, const int line, std::string const& message) {
    log_warn(filename, line, message.c_str());
  }
  inline void log_warn(const char* message) {
    log_warn(__FILE__, __LINE__, message);
  }
  inline void log_warn(std::string const& message) {
    log_warn(__FILE__, __LINE__, message.c_str());
  }


  /** Log message at ERROR level */
  inline void log_error(const char* filename, const int line, const char* message) {
    log(TRITONSERVER_LOG_ERROR, filename, line, message);
  }
  inline void log_error(const char* filename, const int line, std::string const& message) {
    log_error(filename, line, message.c_str());
  }
  inline void log_error(const char* message) {
    log_error(__FILE__, __LINE__, message);
  }
  inline void log_error(std::string const& message) {
    log_error(__FILE__, __LINE__, message.c_str());
  }


  /** Log message at VERBOSE level */
  inline void log_debug(const char* filename, const int line, const char* message) {
    log(TRITONSERVER_LOG_VERBOSE, filename, line, message);
  }
  inline void log_debug(const char* filename, const int line, std::string const& message) {
    log_debug(filename, line, message.c_str());
  }
  inline void log_debug(const char* message) {
    log_debug(__FILE__, __LINE__, message);
  }
  inline void log_debug(std::string const& message) {
    log_debug(__FILE__, __LINE__, message.c_str());
  }

}}}  // namespace triton::backend::rapids
