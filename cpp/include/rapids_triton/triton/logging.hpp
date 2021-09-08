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

#include <ostream>
#include <sstream>
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

  struct log_stream : public std::ostream {
    log_stream(TRITONSERVER_LogLevel level, char const* filename, int line) : std::ostream{},  buffer_{level, filename, line} { rdbuf(&buffer_); }
    log_stream(TRITONSERVER_LogLevel level) : std::ostream{},  buffer_{level, __FILE__, __LINE__} { rdbuf(&buffer_); }

  ~log_stream() {
    try {
      flush();
    } catch (std::ios_base::failure const& ignored_err) {
      // Ignore error if flush fails
    }
  }

   private:
    struct log_buffer : public std::stringbuf {
      log_buffer(TRITONSERVER_LogLevel level, char const* filename, int line) : level_{level}, filename_{filename}, line_{line} {}

      virtual int sync() {
        auto msg = str();
        if(!msg.empty()) {
          log(level_, filename_, line_, msg.c_str());
          str("");
        }
        return 0;
      }

     private:
      TRITONSERVER_LogLevel level_;
      char const* filename_;
      int line_;
    };

    log_buffer buffer_;
  };


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
  inline auto log_info(const char* filename, const int line) {
    return log_stream(TRITONSERVER_LOG_INFO, filename, line);
  }
  inline auto log_info() {
    return log_stream(TRITONSERVER_LOG_INFO);
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
  inline auto log_warn(const char* filename, const int line) {
    return log_stream(TRITONSERVER_LOG_WARN, filename, line);
  }
  inline auto log_warn() {
    return log_stream(TRITONSERVER_LOG_WARN);
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
  inline auto log_error(const char* filename, const int line) {
    return log_stream(TRITONSERVER_LOG_ERROR, filename, line);
  }
  inline auto log_error() {
    return log_stream(TRITONSERVER_LOG_ERROR);
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
  inline auto log_debug(const char* filename, const int line) {
    return log_stream(TRITONSERVER_LOG_VERBOSE, filename, line);
  }
  inline auto log_debug() {
    return log_stream(TRITONSERVER_LOG_VERBOSE);
  }

}}}  // namespace triton::backend::rapids
