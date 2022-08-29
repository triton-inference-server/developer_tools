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
    TRITONSERVER_ModelControlMode* model_control_mode, ModelControlMode mode)
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
WrapperToTritonLogFormat(TRITONSERVER_LogFormat* log_format, LogFormat format)
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

TRITONSERVER_DataType
WrapperToTritonDataType(DataType dtype)
{
  switch (dtype) {
    case BOOL:
      return TRITONSERVER_TYPE_BOOL;
    case UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case UINT64:
      return TRITONSERVER_TYPE_UINT64;
    case INT8:
      return TRITONSERVER_TYPE_INT8;
    case INT16:
      return TRITONSERVER_TYPE_INT16;
    case INT32:
      return TRITONSERVER_TYPE_INT32;
    case INT64:
      return TRITONSERVER_TYPE_INT64;
    case FP16:
      return TRITONSERVER_TYPE_FP16;
    case FP32:
      return TRITONSERVER_TYPE_FP32;
    case FP64:
      return TRITONSERVER_TYPE_FP64;
    case BYTES:
      return TRITONSERVER_TYPE_BYTES;
    case BF16:
      return TRITONSERVER_TYPE_BF16;

    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

DataType
TritonToWrapperDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      return BOOL;
    case TRITONSERVER_TYPE_UINT8:
      return UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return UINT64;
    case TRITONSERVER_TYPE_INT8:
      return INT8;
    case TRITONSERVER_TYPE_INT16:
      return INT16;
    case TRITONSERVER_TYPE_INT32:
      return INT32;
    case TRITONSERVER_TYPE_INT64:
      return INT64;
    case TRITONSERVER_TYPE_FP16:
      return FP16;
    case TRITONSERVER_TYPE_FP32:
      return FP32;
    case TRITONSERVER_TYPE_FP64:
      return FP64;
    case TRITONSERVER_TYPE_BYTES:
      return BYTES;
    case TRITONSERVER_TYPE_BF16:
      return BF16;

    default:
      return INVALID;
  }
}

Error
WrapperToTritonMemoryType(
    TRITONSERVER_MemoryType* memory_type, MemoryType mem_type)
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
    MemoryType* memory_type, TRITONSERVER_MemoryType mem_type)
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
