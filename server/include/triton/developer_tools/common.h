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
#pragma once

namespace triton { namespace developer_tools { namespace server {

//==============================================================================
/// enum classes
///
enum class ModelControlMode { NONE, POLL, EXPLICIT };
enum class MemoryType { CPU, CPU_PINNED, GPU };
enum class DataType {
  INVALID,
  BOOL,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  INT8,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  BYTES,
  BF16
};
enum class ModelReadyState { UNKNOWN, READY, UNAVAILABLE, LOADING, UNLOADING };

//==============================================================================
// TritonException
//
struct TritonException : std::exception {
  TritonException(const std::string& message) : message_(message) {}

  const char* what() const throw() { return message_.c_str(); }

  std::string message_;
};

//==============================================================================
/// Custom Response Allocator Callback function signatures.
///
using ResponseAllocatorAllocFn_t = void (*)(
    const char* tensor_name, size_t byte_size, MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void** buffer,
    MemoryType* actual_memory_type, int64_t* actual_memory_type_id);
using OutputBufferReleaseFn_t = void (*)(
    void* buffer, size_t byte_size, MemoryType memory_type,
    int64_t memory_type_id);
using ResponseAllocatorStartFn_t = void (*)(void* userp);

}}}  // namespace triton::developer_tools::server
