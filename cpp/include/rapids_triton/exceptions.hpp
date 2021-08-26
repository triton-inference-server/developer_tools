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
#include <cuda_runtime_api.h>
#include <exception>
#include <string>

namespace triton { namespace backend { namespace rapids {

using ErrorCode = TRITONSERVER_Error_Code;

namespace Error {
  auto constexpr Unknown = ErrorCode::TRITONSERVER_ERROR_UNKNOWN;
  auto constexpr Internal = ErrorCode::TRITONSERVER_ERROR_INTERNAL;
  auto constexpr NotFound = ErrorCode::TRITONSERVER_ERROR_NOT_FOUND;
  auto constexpr InvalidArg = ErrorCode::TRITONSERVER_ERROR_INVALID_ARG;
  auto constexpr Unavailable = ErrorCode::TRITONSERVER_ERROR_UNAVAILABLE;
  auto constexpr Unsupported = ErrorCode::TRITONSERVER_ERROR_UNSUPPORTED;
  auto constexpr AlreadyExists = ErrorCode::TRITONSERVER_ERROR_ALREADY_EXISTS;
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

  TritonException(ErrorCode code, std::string const & msg)
      : error_(TRITONSERVER_ErrorNew(code, msg.c_str()))
  {
  }

  TritonException(ErrorCode code, char const* msg)
      : error_{TRITONSERVER_ErrorNew(code, msg)}
  {
  }

  TritonException(TRITONSERVER_Error* prev_error) : error_(prev_error) {}

  virtual char const* what() const noexcept
  {
    return TRITONSERVER_ErrorMessage(error_);
  }

  auto* error() { return error_; }

 private:
  TRITONSERVER_Error* error_;
};

inline void triton_check(TRITONSERVER_Error* err) {
  if (err != nullptr) {
    throw TritonException(err);
  }
}

inline void cuda_check(cudaError_t const& err) {
  if (err != cudaSuccess) {
    throw TritonException(Error::Internal, cudaGetErrorString(err));
  }
}

}}}  // namespace triton::backend::rapids

