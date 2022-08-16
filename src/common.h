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

#include <iostream>
#include <list>
#include <string>
#include <vector>
#include "triton/core/tritonserver.h"

namespace triton { namespace high_level { namespace server_api {

enum ModelControlMode {
  MODEL_CONTROL_NONE,
  MODEL_CONTROL_POLL,
  MODEL_CONTROL_EXPLICIT
};

enum LogFormat { LOG_DEFAULT, LOG_ISO8601 };

enum MemoryType { CPU, CPU_PINNED, GPU };

#define FAIL_IF_TRITON_ERR(X, MSG)                                \
  do {                                                            \
    TRITONSERVER_Error* err__ = (X);                              \
    if (err__ != nullptr) {                                       \
      std::cerr << "error: " << (MSG) << ": "                     \
                << TRITONSERVER_ErrorCodeString(err__) << " - "   \
                << TRITONSERVER_ErrorMessage(err__) << std::endl; \
      TRITONSERVER_ErrorDelete(err__);                            \
      exit(1);                                                    \
    }                                                             \
  } while (false)

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    Error err = (X);                                               \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


//==============================================================================
/// Error status reported by client API.
///
class Error {
 public:
  /// Create an error with the specified message.
  /// \param msg The message for the error
  explicit Error(const std::string& msg = "");

  /// Accessor for the message of this error.
  /// \return The messsage for the error. Empty if no error.
  const std::string& Message() const { return msg_; }

  /// Does this error indicate OK status?
  /// \return True if this error indicates "ok"/"success", false if
  /// error indicates a failure.
  bool IsOk() const { return msg_.empty(); }

  /// Convenience "success" value. Can be used as Error::Success to
  /// indicate no error.
  static const Error Success;

 private:
  friend std::ostream& operator<<(std::ostream&, const Error&);
  std::string msg_;
};

//==============================================================================
/// The base class for TritonInferenceServer.
///
class TritonInferenceServer {
 protected:
  // The server object.
  TRITONSERVER_Server* server_ptr_ = nullptr;
  // The allocator object allocating output tensor.
  TRITONSERVER_ResponseAllocator* allocator_ = nullptr;
};

//==============================================================================
/// The base structure for InferenceOptions.
///
struct InferenceOptions {
 protected:
  // Callback functions.
  TRITONSERVER_InferenceRequestReleaseFn_t request_complete;
  TRITONSERVER_InferenceResponseCompleteFn_t response_complete;
};

//==============================================================================
/// An interface for InferInput object to describe the model input for
/// inference.
///
class InferInput {
 public:
  /// Create a InferInput instance that describes a model input.
  /// \param infer_input Returns a new InferInput object.
  /// \param name The name of input whose data will be described by this object.
  /// \param dims The shape of the input.
  /// \param datatype The datatype of the input.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferInput** infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);

  /// Gets name of the associated input tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

  /// Gets datatype of the associated input tensor.
  /// \return The datatype of the tensor.
  const std::string& Datatype() const { return datatype_; }

  /// Gets the shape of the input tensor.
  /// \return The shape of the tensor.
  const std::vector<int64_t>& Shape() const { return shape_; }

  /// Set the shape of input associated with this object.
  /// \param dims the vector of dims representing the new shape
  /// of input.
  /// \return Error object indicating success or failure of the
  /// request.
  Error SetShape(const std::vector<int64_t>& dims);

  /// Prepare this input to receive new tensor values. Forget any
  /// existing values that were set by previous calls to SetSharedMemory()
  /// or AppendRaw().
  /// \return Error object indicating success or failure.
  Error Reset();

  /// Get the data ptr
  /// \return Get the raw pointer.
  void* DataPtr();

  /// Get the total byte size of the tensor.
  uint64_t ByteSize() const;

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  std::string datatype_;

  void* data_ptr_;
  uint64_t byte_size_;

  // For string vector.
  std::list<std::string> str_bufs_;
};

//==============================================================================
/// An InferRequestedOutput object is used to describe the requested model
/// output for inference.
///
class InferRequestedOutput {
 public:
  /// Create a InferRequestedOutput instance that describes a model output being
  /// requested.
  /// \param infer_output Returns a new InferOutputGrpc object.
  /// \param name The name of output being requested.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name);

  /// Gets name of the associated output tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

 private:
  std::string name_;
};

//==============================================================================
/// The base class for InferenceRequest.
///
struct InferenceRequest {
 protected:
  std::vector<InferInput*> inputs = {};
  std::vector<InferRequestedOutput*> outputs = {};
};

//==============================================================================
/// Helper functions.
///
Error ToTritonModelControlMode(
    TRITONSERVER_ModelControlMode* model_control_mode, ModelControlMode mode);
Error ToTritonLogFormat(TRITONSERVER_LogFormat* log_format, LogFormat format);

}}}  // namespace triton::high_level::server_api
