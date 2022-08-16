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
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace high_level { namespace server_api {

class InferResult;
class InferRequest;
class InferOptions;
class Error;
class BufferAllocation;

//==============================================================================
/// Logging options.
///
struct logging_options {
  logging_options()
      : verbose(true), info(false), warn(false), error(false),
        format(LOG_DEFAULT)
  {
  }
  logging_options(
      bool verbose, bool info, bool warn, bool error, LogFormat format)
      : verbose(verbose), info(info), warn(warn), error(error), format(format)
  {
  }

  bool verbose;
  bool info;
  bool warn;
  bool error;
  LogFormat format;
};

//==============================================================================
/// Server parameters.
///
struct ServerParams {
  explicit ServerParams(std::vector<std::string> model_repository_paths)
      : model_repository_paths(model_repository_paths)
  {
    logging = logging_options();
    server_id = "triton";
    backend_dir = "/opt/tritonserver/backends";
    repo_agent_dir = "/opt/tritonserver/repoagents";
    disable_auto_complete_config = false;
    model_control_mode = MODEL_CONTROL_NONE;
  }

  explicit ServerParams(
      std::vector<std::string> model_repository_paths, logging_options logging,
      std::string server_id, std::string backend_dir,
      std::string repo_agent_dir, bool disable_auto_complete_config,
      ModelControlMode model_control_mode)
      : model_repository_paths(model_repository_paths), logging(logging),
        server_id(server_id), backend_dir(backend_dir),
        repo_agent_dir(repo_agent_dir),
        disable_auto_complete_config(disable_auto_complete_config),
        model_control_mode(model_control_mode)
  {
  }

  std::vector<std::string> model_repository_paths;
  logging_options logging;
  std::string server_id;
  std::string backend_dir;
  std::string repo_agent_dir;
  bool disable_auto_complete_config;
  ModelControlMode model_control_mode;
};

//==============================================================================
/// Object that encapsulates in-process C API functionalities.
///
class TritonServer : public TritonInferenceServer {
 public:
  TritonServer(ServerParams server_params);

  ~TritonServer();

  /// Load the requested model or reload the model if it is already loaded.
  /// \param model_name The name of the model.
  /// \return Error object indicating success or failure.
  Error LoadModel(const std::string model_name);

  /// Unload the requested model. Unloading a model that is not loaded
  /// on server has no affect and success code will be returned.
  /// \param model_name The name of the model.
  /// \return Error object indicating success or failure.
  Error UnloadModel(const std::string model_name);

  /// Get the configuration of specified model
  /// \param model_config Returns JSON representation of model configuration
  /// as a string.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model to get configuration.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \return Error object indicating success or failure.
  Error ModelConfig(
      std::string* model_config, const std::string model_name,
      const std::string& model_version = "");

  /// Get the set of names of models that are loaded and ready for inference.
  /// \param loaded_models Returns the set of names of models that are loaded
  /// and ready for inference
  /// \return Error object indicating success or failure.
  Error LoadedModels(std::set<std::string>* loaded_models);

  /// Get the index of model repository contents.
  /// \param repository_index Returns JSON representation of the repository
  /// index as a string.
  /// \return Error object indicating success or failure.
  Error ModelIndex(std::string* repository_index);

  /// Run asynchronous inference on server.
  /// \param infer_result Returns the result of inference as InferResult object.
  /// \param infer_request The InferRequest objects contains the inputs, outputs
  /// and infer options for an inference request.
  /// \return Error object indicating success or failure.
  Error AsyncInfer(
      InferResult** infer_result, const InferRequest infer_request);

  /// Run asynchronous inference on server in-place with poroviding
  /// pre-allocated buffer. \param buffer The pre-allocated output buffer.
  /// \param buffer_byte_size The byte size of the pre-allocated output buffer.
  /// \param infer_request The InferRequest objects contains the inputs, outputs
  /// and infer options for an
  // inference request.
  /// \return Error object indicating success or failure.
  Error AsyncInfer(
      void** buffer, size_t buffer_byte_size, const InferRequest infer_request);

 private:
  std::set<std::string> models_;
};

//==============================================================================
/// Response Allocator Callback function.
///
typedef Error (*ResponseAllocatorCallbackFn_t)(BufferAllocation buffer_alloc);

//==============================================================================
/// Structure to hold options for Inference Request.
///
struct InferOptions : InferenceOptions {
  explicit InferOptions(const std::string& model_name)
      : model_name_(model_name), model_version_(""), request_id_(""),
        sequence_id_(0), sequence_id_str_(""), sequence_start_(false),
        sequence_end_(false), priority_(0), request_timeout_(0),
        alloc_fn(nullptr), release_fn(nullptr), start_fn(nullptr)
  {
  }

  explicit InferOptions(
      const std::string& model_name, const std::string model_version,
      const std::string request_id, const uint64_t sequence_id,
      const std::string sequence_id_str, const bool sequence_start,
      const bool sequence_end, const uint64_t priority,
      const uint64_t request_timeout, ResponseAllocatorCallbackFn_t alloc_fn,
      ResponseAllocatorCallbackFn_t release_fn,
      ResponseAllocatorCallbackFn_t start_fn)
      : model_name_(model_name), model_version_(model_version),
        request_id_(request_id), sequence_id_(sequence_id),
        sequence_id_str_(sequence_id_str), sequence_start_(sequence_start),
        sequence_end_(sequence_end), priority_(priority),
        request_timeout_(request_timeout), alloc_fn(alloc_fn),
        release_fn(release_fn), start_fn(start_fn)
  {
  }

  /// The name of the model to run inference.
  std::string model_name_;
  /// The version of the model to use while running inference. The default
  /// value is an empty string which means the server will select the
  /// version of the model based on its internal policy.
  std::string model_version_;
  /// An identifier for the request. If specified will be returned
  /// in the response. Default value is an empty string which means no
  /// request_id will be used.
  std::string request_id_;
  /// The unique identifier for the sequence being represented by the
  /// object. Default value is 0 which means that the request does not
  /// belong to a sequence. If this value is non-zero, then sequence_id_str_
  /// MUST be set to "".
  uint64_t sequence_id_;
  /// The unique identifier for the sequence being represented by the
  /// object. Default value is "" which means that the request does not
  /// belong to a sequence. If this value is non-empty, then sequence_id_
  /// MUST be set to 0.
  std::string sequence_id_str_;
  /// Indicates whether the request being added marks the start of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_start_;
  /// Indicates whether the request being added marks the end of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_end_;
  /// Indicates the priority of the request. Priority value zero
  /// indicates that the default priority level should be used
  /// (i.e. same behavior as not specifying the priority parameter).
  /// Lower value priorities indicate higher priority levels. Thus
  /// the highest priority level is indicated by setting the parameter
  /// to 1, the next highest is 2, etc. If not provided, the server
  /// will handle the request using default setting for the model.
  uint64_t priority_;
  /// The timeout value for the request, in microseconds. If the request
  /// cannot be completed within the time by the server can take a
  /// model-specific action such as terminating the request. If not
  /// provided, the server will handle the request using default setting
  /// for the model.
  uint64_t request_timeout_;

  /// The custom response allocator for the model. This function is optional. If
  /// not set, will use the provided default allocator
  ResponseAllocatorCallbackFn_t alloc_fn;

  /// The custom response release callback function for the model. This function
  /// is optional. If not set, will use the provided default  response release
  /// callback function
  ResponseAllocatorCallbackFn_t release_fn;

  /// The custom start callback function for the model. This function is
  /// optional. If not set, will not provide any start callback function as it’s
  /// typically not used.
  ResponseAllocatorCallbackFn_t start_fn;
};

//==============================================================================
/// Object that describes an inflight inference request.
///
class InferRequest : public InferenceRequest {
 public:
  InferRequest(InferOptions infer_options){};

  ~InferRequest(){};

  /// Add an input tensor to be sent within an InferRequest object.
  /// \param model_name The name of the input tensor.
  /// \param buffer_ptr The buffer pointer of the input data.
  /// \param byte_size The size, in bytes, of the input data.
  /// \param data_type The type of the input. This is optional.
  /// \param shape The shape of the input. This is optional.
  /// \return Error object indicating success or failure.
  Error AddInput(
      const std::string model_name, char* buffer_ptr, uint64_t byte_size,
      std::string data_type, std::vector<int64_t> shape);

  /// Add an input tensor to be sent within an InferRequest object.
  /// \param model_name The name of the input tensor.
  /// \param begin The begin iterator of the container.
  /// \param end  The end iterator of the container.
  /// \param data_type The data type of the input tensor. This is optional.
  /// \param shape The shape of the input tensor. This is optional.
  /// \return Error object indicating success or failure.
  template <typename Iterator>
  Error AddInput(
      const std::string model_name, const Iterator& begin, const Iterator& end,
      std::string data_type, std::vector<int64_t> shape);

  /// Set the memory type and memory type id of input.
  /// \param input_memory_type The input memory type.
  /// \param intput_memory_type_id The input memory typ ide.
  /// \return Error object indicating success or failure.
  Error SetInputMemoryTypeAndId(
      const MemoryType input_memory_type, const int64_t intput_memory_type_id);

  /// Add an requested output tensor to be sent within an InferRequest object.
  /// \param name The name of the requested output tensor.
  /// \return Error object indicating success or failure.
  Error AddRequestedOutputName(const std::string name);

  /// Clear inputs and outputs of the request except for the callback functions.
  /// \return Error object indicating success or failure.
  Error Reset();

 private:
  MemoryType input_memory_type;
  int64_t intput_memory_type_id;
};

//==============================================================================
/// An interface for InferResult object to interpret the response to an
/// inference request.
///
class InferResult {
 public:
  InferResult(){};
  ~InferResult();

  /// Get the name of the model which generated this response.
  /// \param name Returns the name of the model.
  /// \return Error object indicating success or failure.
  Error ModelName(std::string* name);

  /// Get the version of the model which generated this response.
  /// \param version Returns the version of the model.
  /// \return Error object indicating success or failure.
  Error ModelVersion(std::string* version);

  /// Get the id of the request which generated this response.
  /// \param version Returns the version of the model.
  /// \return Error object indicating success or failure.
  Error Id(std::string* id);

  /// Get the shape of output result returned in the response.
  /// \param output_name The name of the ouput to get shape.
  /// \param shape Returns the shape of result for specified output name.
  /// \return Error object indicating success or failure.
  Error Shape(const std::string& output_name, std::vector<int64_t>* shape);

  /// Get the datatype of output result returned in the response.
  /// \param output_name The name of the ouput to get datatype.
  /// \param shape Returns the datatype of result for specified output name.
  /// \return Error object indicating success or failure.
  Error Datatype(const std::string& output_name, std::string* datatype);

  /// Get access to the buffer holding raw results of specified output
  /// returned by the server. Note the buffer is owned by InferResult
  /// instance. Users can copy out the data if required to extend the
  /// lifetime.
  /// \param output_name The name of the output to get result data.
  /// \param buf Returns the pointer to the start of the buffer.
  /// \param byte_size Returns the size of buffer in bytes.
  /// \return Error object indicating success or failure of the
  /// request.
  Error RawData(
      const std::string& output_name, const uint8_t** buf, size_t* byte_size);

  /// Get the result data as a vector of strings. The vector will
  /// receive a copy of result data. An error will be generated if
  /// the datatype of output is not 'BYTES'.
  /// \param output_name The name of the output to get result data.
  /// \param string_result Returns the result data represented as
  /// a vector of strings. The strings are stored in the
  /// row-major order.
  /// \return Error object indicating success or failure of the
  /// request.
  Error StringData(
      const std::string& output_name, std::vector<std::string>* string_result);

  /// Returns the complete response as a user friendly string.
  /// \return The string describing the complete response.
  std::string DebugString();
};

//==============================================================================
/// BufferAllocation
///
class BufferAllocation {
  struct BufferProperties {
    const char* tensor_name;

    size_t byte_size;

    const MemoryType memory_type;

    int64_t memory_type_id;

    void* userp;

    void** buffer;

    void** buffer_userp;
  };

  BufferAllocation(struct BufferProperties);

  // Custom allocation here…
  void Allocate();
};

}}}  // namespace triton::high_level::server_api
