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

namespace triton { namespace triton_developer_tools { namespace server {

Error
ToTritonModelControlMode(
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
ToTritonLogFormat(TRITONSERVER_LogFormat* log_format, LogFormat format)
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
ToTritonDataType(TRITONSERVER_DataType* dtype, std::string data_type)
{
  if ((data_type == "BOOL") || (data_type == "TYPE_BOOL")) {
    *dtype = TRITONSERVER_TYPE_BOOL;
  } else if ((data_type == "UINT8") || (data_type == "TYPE_UINT8")) {
    *dtype = TRITONSERVER_TYPE_UINT8;
  } else if ((data_type == "UINT16") || (data_type == "TYPE_UINT16")) {
    *dtype = TRITONSERVER_TYPE_UINT16;
  } else if ((data_type == "UINT32") || (data_type == "TYPE_UINT32")) {
    *dtype = TRITONSERVER_TYPE_UINT32;
  } else if ((data_type == "UINT64") || (data_type == "TYPE_UINT64")) {
    *dtype = TRITONSERVER_TYPE_UINT64;
  } else if ((data_type == "INT8") || (data_type == "TYPE_INT8")) {
    *dtype = TRITONSERVER_TYPE_INT8;
  } else if ((data_type == "INT16") || (data_type == "TYPE_INT16")) {
    *dtype = TRITONSERVER_TYPE_INT16;
  } else if ((data_type == "INT32") || (data_type == "TYPE_INT32")) {
    *dtype = TRITONSERVER_TYPE_INT32;
  } else if ((data_type == "INT64") || (data_type == "TYPE_INT64")) {
    *dtype = TRITONSERVER_TYPE_INT64;
  } else if ((data_type == "FP16") || (data_type == "TYPE_FP16")) {
    *dtype = TRITONSERVER_TYPE_FP16;
  } else if ((data_type == "FP32") || (data_type == "TYPE_FP32")) {
    *dtype = TRITONSERVER_TYPE_FP32;
  } else if ((data_type == "FP64") || (data_type == "TYPE_FP64")) {
    *dtype = TRITONSERVER_TYPE_FP64;
  } else if ((data_type == "BYTES") || (data_type == "TYPE_STRING")) {
    *dtype = TRITONSERVER_TYPE_BYTES;
  } else if ((data_type == "BF16") || (data_type == "TYPE_BF16")) {
    *dtype = TRITONSERVER_TYPE_BF16;
  } else {
    *dtype = TRITONSERVER_TYPE_INVALID;
  }

  return Error::Success;
}

Error
ToTritonMemoryType(TRITONSERVER_MemoryType* memory_type, MemoryType mem_type)
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
ToMemoryType(MemoryType* memory_type, TRITONSERVER_MemoryType mem_type)
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

std::string
MemoryTypeString(MemoryType memory_type)
{
  switch (memory_type) {
    case CPU:
      return "CPU";
    case CPU_PINNED:
      return "CPU_PINNED";
    case GPU:
      return "GPU";
    default:
      break;
  }

  return "<invalid>";
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

}}}  // namespace triton::triton_developer_tools::server
