// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "common.h"

namespace triton { namespace server { namespace wrapper {

Error
WrapperToTritonModelControlMode(
    TRITONSERVER_ModelControlMode* model_control_mode,
    Wrapper_ModelControlMode mode)
{
  switch (mode) {
    case MODEL_CONTROL_NONE:
      *model_control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
      break;
    case MODEL_CONTROL_POLL:
      *model_control_mode = TRITONSERVER_MODEL_CONTROL_POLL;
      break;
    case MODEL_CONTROL_EXPLICIT:
      *model_control_mode = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
      break;

    default:
      return Error("unsupported model control mode.");
  }

  return Error::Success;
}

Error
WrapperToTritonLogFormat(
    TRITONSERVER_LogFormat* log_format, Wrapper_LogFormat format)
{
  switch (format) {
    case LOG_DEFAULT:
      *log_format = TRITONSERVER_LOG_DEFAULT;
      break;
    case LOG_ISO8601:
      *log_format = TRITONSERVER_LOG_ISO8601;
      break;

    default:
      return Error("unsupported log format.");
  }

  return Error::Success;
}

Error
WrapperToTritonDataType(
    TRITONSERVER_DataType* data_type, Wrapper_DataType dtype)
{
  switch (dtype) {
    case BOOL:
      *data_type = TRITONSERVER_TYPE_BOOL;
      break;
    case UINT8:
      *data_type = TRITONSERVER_TYPE_UINT8;
      break;
    case UINT16:
      *data_type = TRITONSERVER_TYPE_UINT16;
      break;
    case UINT32:
      *data_type = TRITONSERVER_TYPE_UINT32;
      break;
    case UINT64:
      *data_type = TRITONSERVER_TYPE_UINT64;
      break;
    case INT8:
      *data_type = TRITONSERVER_TYPE_INT8;
      break;
    case INT16:
      *data_type = TRITONSERVER_TYPE_INT16;
      break;
    case INT32:
      *data_type = TRITONSERVER_TYPE_INT32;
      break;
    case INT64:
      *data_type = TRITONSERVER_TYPE_INT64;
      break;
    case FP16:
      *data_type = TRITONSERVER_TYPE_FP16;
      break;
    case FP32:
      *data_type = TRITONSERVER_TYPE_FP32;
      break;
    case FP64:
      *data_type = TRITONSERVER_TYPE_FP64;
      break;
    case BYTES:
      *data_type = TRITONSERVER_TYPE_BYTES;
      break;
    case BF16:
      *data_type = TRITONSERVER_TYPE_BF16;
      break;

    default:
      *data_type = TRITONSERVER_TYPE_INVALID;
      break;
  }

  return Error::Success;
}

Error
TritonToWrapperDataType(
    Wrapper_DataType* data_type, TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      *data_type = BOOL;
      break;
    case TRITONSERVER_TYPE_UINT8:
      *data_type = UINT8;
      break;
    case TRITONSERVER_TYPE_UINT16:
      *data_type = UINT16;
      break;
    case TRITONSERVER_TYPE_UINT32:
      *data_type = UINT32;
      break;
    case TRITONSERVER_TYPE_UINT64:
      *data_type = UINT64;
      break;
    case TRITONSERVER_TYPE_INT8:
      *data_type = INT8;
      break;
    case TRITONSERVER_TYPE_INT16:
      *data_type = INT16;
      break;
    case TRITONSERVER_TYPE_INT32:
      *data_type = INT32;
      break;
    case TRITONSERVER_TYPE_INT64:
      *data_type = INT64;
      break;
    case TRITONSERVER_TYPE_FP16:
      *data_type = FP16;
      break;
    case TRITONSERVER_TYPE_FP32:
      *data_type = FP32;
      break;
    case TRITONSERVER_TYPE_FP64:
      *data_type = FP64;
      break;
    case TRITONSERVER_TYPE_BYTES:
      *data_type = BYTES;
      break;
    case TRITONSERVER_TYPE_BF16:
      *data_type = BF16;
      break;

    default:
      *data_type = INVALID;
      break;
  }

  return Error::Success;
}

Error
WrapperToTritonMemoryType(
    TRITONSERVER_MemoryType* memory_type, Wrapper_MemoryType mem_type)
{
  switch (mem_type) {
    case CPU:
      *memory_type = TRITONSERVER_MEMORY_CPU;
      break;
    case CPU_PINNED:
      *memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
      break;
    case GPU:
      *memory_type = TRITONSERVER_MEMORY_GPU;
      break;

    default:
      return Error("unsupported memory type.");
  }

  return Error::Success;
}

Error
TritonToWrapperMemoryType(
    Wrapper_MemoryType* memory_type, TRITONSERVER_MemoryType mem_type)
{
  switch (mem_type) {
    case TRITONSERVER_MEMORY_CPU:
      *memory_type = CPU;
      break;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      *memory_type = CPU_PINNED;
      break;
    case TRITONSERVER_MEMORY_GPU:
      *memory_type = GPU;
      break;

    default:
      return Error("unsupported memory type.");
  }

  return Error::Success;
}

const Error Error::Success("");

Error::Error(const std::string& msg) : msg_(msg) {}

std::ostream&
operator<<(std::ostream& out, const Error& err)
{
  if (!err.msg_.empty()) {
    out << err.msg_;
  }
  return out;
}

}}}  // namespace triton::server::wrapper
