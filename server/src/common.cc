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

namespace tsw = triton::server::wrapper;

TRITONSERVER_ModelControlMode
ToTritonModelControlMode(const ModelControlMode& mode)
{
  switch (mode) {
    case tsw::ModelControlMode::MODEL_CONTROL_NONE:
      return TRITONSERVER_MODEL_CONTROL_NONE;
    case tsw::ModelControlMode::MODEL_CONTROL_POLL:
      return TRITONSERVER_MODEL_CONTROL_POLL;
    case tsw::ModelControlMode::MODEL_CONTROL_EXPLICIT:
      return TRITONSERVER_MODEL_CONTROL_EXPLICIT;

    default:
      throw TritonException("unsupported model control mode.");
  }
}

TRITONSERVER_LogFormat
ToTritonLogFormat(const LogFormat& format)
{
  switch (format) {
    case tsw::LogFormat::LOG_DEFAULT:
      return TRITONSERVER_LOG_DEFAULT;
    case tsw::LogFormat::LOG_ISO8601:
      return TRITONSERVER_LOG_ISO8601;

    default:
      throw TritonException("unsupported log format.");
  }
}

TRITONSERVER_DataType
ToTritonDataType(const DataType& dtype) noexcept
{
  switch (dtype) {
    case tsw::DataType::BOOL:
      return TRITONSERVER_TYPE_BOOL;
    case tsw::DataType::UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case tsw::DataType::UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case tsw::DataType::UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case tsw::DataType::UINT64:
      return TRITONSERVER_TYPE_UINT64;
    case tsw::DataType::INT8:
      return TRITONSERVER_TYPE_INT8;
    case tsw::DataType::INT16:
      return TRITONSERVER_TYPE_INT16;
    case tsw::DataType::INT32:
      return TRITONSERVER_TYPE_INT32;
    case tsw::DataType::INT64:
      return TRITONSERVER_TYPE_INT64;
    case tsw::DataType::FP16:
      return TRITONSERVER_TYPE_FP16;
    case tsw::DataType::FP32:
      return TRITONSERVER_TYPE_FP32;
    case tsw::DataType::FP64:
      return TRITONSERVER_TYPE_FP64;
    case tsw::DataType::BYTES:
      return TRITONSERVER_TYPE_BYTES;
    case tsw::DataType::BF16:
      return TRITONSERVER_TYPE_BF16;

    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

DataType
TritonToDataType(const TRITONSERVER_DataType& dtype) noexcept
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      return tsw::DataType::BOOL;
    case TRITONSERVER_TYPE_UINT8:
      return tsw::DataType::UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return tsw::DataType::UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return tsw::DataType::UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return tsw::DataType::UINT64;
    case TRITONSERVER_TYPE_INT8:
      return tsw::DataType::INT8;
    case TRITONSERVER_TYPE_INT16:
      return tsw::DataType::INT16;
    case TRITONSERVER_TYPE_INT32:
      return tsw::DataType::INT32;
    case TRITONSERVER_TYPE_INT64:
      return tsw::DataType::INT64;
    case TRITONSERVER_TYPE_FP16:
      return tsw::DataType::FP16;
    case TRITONSERVER_TYPE_FP32:
      return tsw::DataType::FP32;
    case TRITONSERVER_TYPE_FP64:
      return tsw::DataType::FP64;
    case TRITONSERVER_TYPE_BYTES:
      return tsw::DataType::BYTES;
    case TRITONSERVER_TYPE_BF16:
      return tsw::DataType::BF16;

    default:
      return tsw::DataType::INVALID;
  }
}

TRITONSERVER_MemoryType
ToTritonMemoryType(const MemoryType& mem_type)
{
  switch (mem_type) {
    case tsw::MemoryType::CPU:
      return TRITONSERVER_MEMORY_CPU;
    case tsw::MemoryType::CPU_PINNED:
      return TRITONSERVER_MEMORY_CPU_PINNED;
    case tsw::MemoryType::GPU:
      return TRITONSERVER_MEMORY_GPU;

    default:
      throw TritonException("unsupported memory type.");
  }
}

MemoryType
TritonToMemoryType(const TRITONSERVER_MemoryType& mem_type)
{
  switch (mem_type) {
    case TRITONSERVER_MEMORY_CPU:
      return tsw::MemoryType::CPU;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return tsw::MemoryType::CPU_PINNED;
    case TRITONSERVER_MEMORY_GPU:
      return tsw::MemoryType::GPU;

    default:
      throw TritonException("unsupported memory type.");
  }
}

ModelReadyState
StringToModelReadyState(const std::string& state) noexcept
{
  if (state == "UNKNOWN") {
    return tsw::ModelReadyState::UNKNOWN;
  } else if (state == "READY") {
    return tsw::ModelReadyState::READY;
  } else if (state == "UNAVAILABLE") {
    return tsw::ModelReadyState::UNAVAILABLE;
  } else if (state == "LOADING") {
    return tsw::ModelReadyState::LOADING;
  } else if (state == "UNLOADING") {
    return tsw::ModelReadyState::UNLOADING;
  } else {
    return tsw::ModelReadyState::UNKNOWN;
  }
}

}}}  // namespace triton::server::wrapper
