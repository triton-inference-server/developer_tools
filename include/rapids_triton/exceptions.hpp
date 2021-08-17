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
#include <triton/core/tritonserver.h>
#include <exception>
#include <string>

namespace triton { namespace backend { namespace rapids {

using ErrorCode = TRITONSERVER_Error_Code;

namespace Error {
  using Unknown = ErrorCode::TRITONSERVER_ERROR_UNKNOWN;
  using Internal = ErrorCode::TRITONSERVER_ERROR_INTERNAL;
  using NotFound = ErrorCode::TRITONSERVER_ERROR_NOT_FOUND;
  using InvalidArg = ErrorCode::TRITONSERVER_ERROR_INVALID_ARG;
  using Unavailable = ErrorCode::TRITONSERVER_ERROR_UNAVAILABLE;
  using Unsupported = ErrorCode::TRITONSERVER_ERROR_UNSUPPORTED;
  using AlreadyExists = ErrorCode::TRITONSERVER_ERROR_ALREADY_EXISTS;
}

/**
 * @brief Exception thrown if processing cannot continue for a request
 *
 * This exception should be thrown whenever a condition is encountered that (if
 * it is not appropriately handled by some other exception handler) SHOULD
 * result in Triton reporting an error for the request being processed. It
 * signals that (absent any other fallbacks), this request cannot be fulfilled
 * but that the server may still be in a state to continue handling other
 * requests, including requests to other models.
 */
struct TritonException : std::exception {

 public:
  TritonException()
      : error_(TRITONSERVER_ErrorNew(Error::Unknown,
            "encountered unknown error"))
  {
  }

  TritonException(ErrorCode code, const std::string& msg)
      : error_(TRITONSERVER_ErrorNew(code, msg.c_str()))
  {
  }

  TritonException(ErrorCode code, const char* msg)
      : error_{TRITONSERVER_ErrorNew(code, msg)}
  {
  }

  // Exists only for triton_check; should not be used elsewhere
  TritonException(TRITONSERVER_Error* prev_error) : error_(prev_error) {}

  const char* what() const noexcept
  {
    return TRITONSERVER_ErrorMessage(error_);
  }

  TRITONSERVER_Error* error() { return error_; }

 private:
  TRITONSERVER_Error* error_;
};

inline void triton_check(TRITONSERVER_Error* err) {
  if (err != nullptr) {
    throw TritonException(err);
  }
}

}}}  // namespace triton::backend::rapids

