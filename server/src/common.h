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
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace server { namespace wrapper {

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
#define RETURN_ERR_IF_TRITON_ERR(X)                                \
  do {                                                             \
    TRITONSERVER_Error* err__ = (X);                               \
    if (err__ != nullptr) {                                        \
      return Error(                                                \
          TRITONSERVER_ErrorCodeString(err__) + std::string("-") + \
          TRITONSERVER_ErrorMessage(err__) + "\n");                \
    }                                                              \
  } while (false)
#define IGNORE_ERROR(X)                   \
  do {                                    \
    TRITONSERVER_Error* ie_err__ = (X);   \
    if (ie_err__ != nullptr) {            \
      TRITONSERVER_ErrorDelete(ie_err__); \
    }                                     \
  } while (false)
#define RETURN_IF_ERR(X)           \
  {                                \
    Error err = (X);               \
    if (!err.IsOk()) {             \
      return Error(err.Message()); \
    }                              \
  }
#define LOG_IF_ERROR(X, MSG)                                                   \
  do {                                                                         \
    TRITONSERVER_Error* lie_err__ = (X);                                       \
    if (lie_err__ != nullptr) {                                                \
      IGNORE_ERROR(TRITONSERVER_LogMessage(                                    \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                           \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(lie_err__) + \
           " - " + TRITONSERVER_ErrorMessage(lie_err__))                       \
              .c_str()));                                                      \
      TRITONSERVER_ErrorDelete(lie_err__);                                     \
    }                                                                          \
  } while (false)
#define LOG_MESSAGE(LEVEL, MSG)                                  \
  do {                                                           \
    LOG_IF_ERROR(                                                \
        TRITONSERVER_LogMessage(LEVEL, __FILE__, __LINE__, MSG), \
        ("failed to log message: "));                            \
  } while (false)
#define THROW_IF_ERR(X)               \
  do {                                \
    Error err = (X);                  \
    if (!err.IsOk()) {                \
      throw Exception(err.Message()); \
    }                                 \
  } while (false)
#define THROW_IF_TRITON_ERR(X)                                     \
  do {                                                             \
    TRITONSERVER_Error* err__ = (X);                               \
    if (err__ != nullptr) {                                        \
      Exception ex(                                                \
          TRITONSERVER_ErrorCodeString(err__) + std::string("-") + \
          TRITONSERVER_ErrorMessage(err__) + "\n");                \
      TRITONSERVER_ErrorDelete(err__);                             \
      throw ex;                                                    \
    }                                                              \
  } while (false)
#define THROW_ERR_IF_TRITON_ERR(X)                                 \
  do {                                                             \
    TRITONSERVER_Error* err__ = (X);                               \
    if (err__ != nullptr) {                                        \
      Error err = Error(                                           \
          TRITONSERVER_ErrorCodeString(err__) + std::string("-") + \
          TRITONSERVER_ErrorMessage(err__) + "\n");                \
      TRITONSERVER_ErrorDelete(err__);                             \
      throw Exception(err.Message());                              \
    }                                                              \
  } while (false)
#define RETURN_TRITON_ERR_IF_ERR(X)                              \
  do {                                                           \
    Error err = (X);                                             \
    if (!err.IsOk()) {                                           \
      return TRITONSERVER_ErrorNew(                              \
          TRITONSERVER_ERROR_INTERNAL, (err.Message()).c_str()); \
    }                                                            \
  } while (false)

//==============================================================================
enum Wrapper_ModelControlMode {
  MODEL_CONTROL_NONE,
  MODEL_CONTROL_POLL,
  MODEL_CONTROL_EXPLICIT
};
enum Wrapper_MemoryType { CPU, CPU_PINNED, GPU };
enum Wrapper_LogFormat { LOG_DEFAULT, LOG_ISO8601 };
enum Wrapper_DataType {
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

//==============================================================================
// Exception
//
struct Exception : std::exception {
  Exception(const std::string& message) : message_(message) {}

  const char* what() const throw() { return message_.c_str(); }

  std::string message_;
};

//==============================================================================
/// Structure to hold response parameters for InfeResult object.
///
struct ResponseParameters {
  explicit ResponseParameters(
      const char* name, TRITONSERVER_ParameterType type, const void* vvalue)
      : name_(name), type_(type), vvalue_(vvalue)
  {
  }

  const char* name_;
  TRITONSERVER_ParameterType type_;
  const void* vvalue_;
};

//==============================================================================
/// Error status reported by server C++ API.
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
/// Custom Response Allocator Callback function signatures.
///
typedef Error (*ResponseAllocatorAllocFn_t)(
    const char* tensor_name, size_t byte_size,
    Wrapper_MemoryType preferred_memory_type, int64_t preferred_memory_type_id,
    void* userp, void** buffer, void** buffer_userp,
    Wrapper_MemoryType* actual_memory_type, int64_t* actual_memory_type_id);

typedef Error (*ResponseAllocatorReleaseFn_t)(
    void* buffer, void* buffer_userp, size_t byte_size,
    Wrapper_MemoryType memory_type, int64_t memory_type_id);

typedef Error (*ResponseAllocatorStartFn_t)(void* userp);

//==============================================================================
/// Helper functions.
///
Error WrapperToTritonModelControlMode(
    TRITONSERVER_ModelControlMode* model_control_mode,
    Wrapper_ModelControlMode mode);
Error WrapperToTritonLogFormat(
    TRITONSERVER_LogFormat* log_format, Wrapper_LogFormat format);
Error WrapperToTritonDataType(
    TRITONSERVER_DataType* data_type, Wrapper_DataType dtype);
Error TritonToWrapperDataType(
    Wrapper_DataType* data_type, TRITONSERVER_DataType dtype);
Error WrapperToTritonMemoryType(
    TRITONSERVER_MemoryType* memory_type, Wrapper_MemoryType mem_type);
Error TritonToWrapperMemoryType(
    Wrapper_MemoryType* memory_type, TRITONSERVER_MemoryType mem_type);

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
  /// \param data_ptr The data pointer of the input.
  /// \param byte_size The byte size of the input.
  /// \param memory_type The memory type of the input.
  /// \param memory_type_id The memory type id of the input.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferInput** infer_input, const std::string name,
      const std::vector<int64_t>& dims, const Wrapper_DataType datatype,
      const char* data_ptr, const uint64_t byte_size,
      const Wrapper_MemoryType memory_type, const int64_t memory_type_id)
  {
    TRITONSERVER_MemoryType input_memory_type;
    TRITONSERVER_DataType dtype;
    RETURN_IF_ERR(WrapperToTritonMemoryType(&input_memory_type, memory_type));
    RETURN_IF_ERR(WrapperToTritonDataType(&dtype, datatype));

    *infer_input = new InferInput(
        name, dims, dtype, data_ptr, byte_size, input_memory_type,
        memory_type_id);
    return Error::Success;
  }

  /// Gets name of the associated input tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

  /// Gets datatype of the associated input tensor.
  /// \return The datatype of the tensor.
  const TRITONSERVER_DataType& DataType() const { return datatype_; }

  /// Gets the shape of the input tensor.
  /// \return The shape of the tensor.
  const std::vector<int64_t>& Shape() const { return shape_; }

  /// Gets the memory type of the input tensor.
  /// \return The memory type of the tensor.
  const TRITONSERVER_MemoryType& MemoryType() const { return memory_type_; }

  /// Gets the memory type id of the input tensor.
  /// \return The memory type id of the tensor.
  const int64_t& MemoryTypeId() const { return memory_type_id_; }

  /// Get the data ptr
  /// \return Get the raw pointer.
  const char* DataPtr() { return data_ptr_; };

  /// Get the total byte size of the tensor.
  uint64_t ByteSize() const { return byte_size_; };

  InferInput(
      const std::string name, const std::vector<int64_t>& shape,
      const TRITONSERVER_DataType datatype, const char* data_ptr,
      const uint64_t byte_size, const TRITONSERVER_MemoryType memory_type,
      const int64_t memory_type_id)
      : name_(name), shape_(shape), datatype_(datatype), data_ptr_(data_ptr),
        byte_size_(byte_size), memory_type_(memory_type),
        memory_type_id_(memory_type_id)
  {
  }


 private:
  std::string name_;
  std::vector<int64_t> shape_;
  TRITONSERVER_DataType datatype_;

  const char* data_ptr_;
  uint64_t byte_size_;

  TRITONSERVER_MemoryType memory_type_;
  int64_t memory_type_id_;
};

//==============================================================================
/// An InferRequestedOutput object is used to describe the requested model
/// output for inference.
///
class InferRequestedOutput {
 public:
  /// Create a InferRequestedOutput instance that describes a model output being
  /// requested.
  /// \param infer_output Returns a new InferRequestedOutput object.
  /// \param name The name of output being requested.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name)
  {
    *infer_output = new InferRequestedOutput(name);
    return Error::Success;
  }

  /// Create a InferRequestedOutput instance that describes a model output being
  /// requested with pre-allocated output buffer.
  /// \param infer_output Returns a new InferRequestedOutput object.
  /// \param name The name of output being requested.
  /// \param buffer The pointer to the start of the pre-allocated buffer.
  /// \param byte_size The size of buffer in bytes.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name,
      const char* buffer, size_t byte_size)
  {
    *infer_output = new InferRequestedOutput(name, buffer, byte_size);
    return Error::Success;
  }

  /// Gets name of the associated output tensor.
  /// \return The name of the tensor.
  const std::string& Name() const { return name_; }

  /// Gets buffer of the associated output tensor.
  /// \return The name of the tensor.
  const char* Buffer() { return buffer_; }

  /// Gets byte size of the associated output tensor.
  /// \return The name of the tensor.
  size_t ByteSize() { return byte_size_; }

  InferRequestedOutput(const std::string& name)
      : name_(name), buffer_(nullptr), byte_size_(0)
  {
  }

  InferRequestedOutput(
      const std::string& name, const char* buffer, size_t byte_size)
      : name_(name), buffer_(buffer), byte_size_(byte_size)
  {
  }

 private:
  std::string name_;
  const char* buffer_;
  size_t byte_size_;
};

//==============================================================================
/// An interface for InferOutput object to describe the infer output for
/// inference.
///
class InferOutput {
 public:
  /// Create a InferOutput instance that describes a inference output.
  /// \param infer_output Returns a new InferOutput object.
  /// \param name The name of output being requested.
  /// \param datatype The datatype of the output.
  /// \param shape The shape of the output.
  /// \param dims_count The dims count of the output.
  /// \param byte_size The byte size of the output.
  /// \param memory_type The memory type of the output.
  /// \param memory_type_id The memory type id of the output.
  /// \param base The data pointer of the output.
  /// \param userp The userp function.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferOutput** infer_output, const char* name,
      TRITONSERVER_DataType data_type, const int64_t* shape,
      uint64_t dims_count, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      const void* base, void* userp)
  {
    *infer_output = new InferOutput(
        name, data_type, shape, dims_count, byte_size, memory_type,
        memory_type_id, base, userp);

    return Error::Success;
  }

  InferOutput(
      const char* name, TRITONSERVER_DataType data_type, const int64_t* shape,
      uint64_t dims_count, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      const void* base, void* userp)
      : name_(name), datatype(data_type), shape(shape), dims_count_(dims_count),
        output_byte_size_(byte_size), memory_type_(memory_type),
        memory_type_id_(memory_type_id), base_(base), userp_(userp)
  {
  }

  /// Gets name of the associated output tensor.
  /// \return The name of the tensor.
  const char* Name() { return name_; }

  /// Gets datatype of the associated output tensor.
  /// \return The datatype of the tensor.
  TRITONSERVER_DataType DataType() { return datatype; }

  /// Gets the shape of the output tensor.
  /// \return The shape of the tensor.
  const int64_t* Shape() { return shape; }

  /// Gets dims count of the associated output tensor.
  /// \return The dims count of the tensor.
  uint64_t DimsCount() { return dims_count_; }

  /// Gets byte size of the associated output tensor.
  /// \return The name of the tensor.
  size_t ByteSize() { return output_byte_size_; }

  /// Gets the memory type of the output tensor.
  /// \return The memory type of the tensor.
  TRITONSERVER_MemoryType MemoryType() { return memory_type_; }

  /// Gets the memory type id of the output tensor.
  /// \return The memory type id of the tensor.
  int64_t MemoryTypeId() { return memory_type_id_; }

  /// Gets data pointer of the associated output tensor.
  /// \return The name of the tensor.
  const void* DataPtr() { return base_; }

 private:
  const char* name_;
  TRITONSERVER_DataType datatype;
  const int64_t* shape;
  uint64_t dims_count_;
  size_t output_byte_size_;
  TRITONSERVER_MemoryType memory_type_;
  int64_t memory_type_id_;
  const void* base_;
  void* userp_;
};

}}}  // namespace triton::server::wrapper
