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

#include <string>
#include "../include/triton/developer_tools/common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace developer_tools { namespace server {

//==============================================================================
/// An InferRequestedOutput object is used to describe the requested model
/// output for inference.
///
class InferRequestedOutput {
 public:
  /// Create an InferRequestedOutput instance that describes a model output
  /// being requested.
  /// \param name The name of output being requested.
  /// \return Returns a new InferRequestedOutput object.
  static std::unique_ptr<InferRequestedOutput> Create(const std::string& name)
  {
    return std::unique_ptr<InferRequestedOutput>(
        new InferRequestedOutput(name));
  }

  /// Create a InferRequestedOutput instance that describes a model output being
  /// requested with pre-allocated output buffer.
  /// \param name The name of output being requested.
  /// \param buffer The pointer to the start of the pre-allocated buffer.
  /// \param byte_size The size of buffer in bytes.
  /// \param memory_type The memory type of the output.
  /// \param memory_type_id The memory type id of the output.
  /// \return Returns a new InferRequestedOutput object.
  static std::unique_ptr<InferRequestedOutput> Create(
      const std::string& name, const char* buffer, size_t byte_size,
      MemoryType memory_type, int64_t memory_type_id)
  {
    return std::unique_ptr<InferRequestedOutput>(new InferRequestedOutput(
        name, buffer, byte_size, memory_type, memory_type_id));
  }

  /// Get name of the associated output tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

  /// Get buffer of the associated output tensor.
  /// \return The name of the tensor.
  const char* Buffer() { return buffer_; }

  /// Get byte size of the associated output tensor.
  /// \return The name of the tensor.
  size_t ByteSize() { return byte_size_; }

  /// Get the memory type of the output tensor.
  /// \return The memory type of the tensor.
  const MemoryType& GetMemoryType() const { return memory_type_; }

  /// Get the memory type id of the output tensor.
  /// \return The memory type id of the tensor.
  const int64_t& MemoryTypeId() const { return memory_type_id_; }

  InferRequestedOutput(const std::string& name)
      : name_(name), buffer_(nullptr), byte_size_(0),
        memory_type_(MemoryType::CPU), memory_type_id_(0)
  {
  }

  InferRequestedOutput(
      const std::string& name, const char* buffer, size_t byte_size,
      MemoryType memory_type, int64_t memory_type_id)
      : name_(name), buffer_(buffer), byte_size_(byte_size),
        memory_type_(memory_type), memory_type_id_(memory_type_id)
  {
  }

 private:
  std::string name_;
  const char* buffer_;
  size_t byte_size_;
  MemoryType memory_type_;
  int64_t memory_type_id_;
};

}}}  // namespace triton::developer_tools::server
