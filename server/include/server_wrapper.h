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

#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "../src/common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server { namespace wrapper {

class InferResult;
class InferRequest;
class Allocator;

//==============================================================================
/// Structure to hold logging options for server parameters.
///
struct LoggingOptions {
  LoggingOptions();

  LoggingOptions(
      bool verbose, bool info, bool warn, bool error, LogFormat format,
      std::string log_file);

  // Verbose logging level. Default is 0.
  uint verbose_;
  // Enable or disable info logging level. Default is true.
  bool info_;
  // Enable or disable warn logging level. Default is true.
  bool warn_;
  // Enable or disable error logging level. Default is true.
  bool error_;
  // The format of logging. For "LOG_DEFAULT", the log severity (L) and
  // timestamp will be logged as "LMMDD hh:mm:ss.ssssss". For "LOG_ISO8601", the
  // log format will be "YYYY-MM-DDThh:mm:ssZ L". Default is 'LOG_DEFAULT'.
  LogFormat format_;
  // Logging output file. If specified, log outputs will be saved to this file.
  // If not specified, log outputs will stream to the console. Default is an
  // empty string.
  std::string log_file_;  // logging output file
};

//==============================================================================
/// Structure to hold metrics options for server parameters.
///
struct MetricsOptions {
  MetricsOptions();

  MetricsOptions(
      bool allow_metrics, bool allow_gpu_metrics, uint64_t metrics_interval_ms);

  // Enable or disable metrics. Default is true.
  bool allow_metrics_;
  // Enable or disable GPU metrics. Default is true.
  bool allow_gpu_metrics_;
  // The interval for metrics collection. Default is 2000.
  uint64_t metrics_interval_ms_;
};

//==============================================================================
/// Structure to hold backend configuration for server parameters.
///
struct BackendConfig {
  BackendConfig();

  BackendConfig(
      std::string backend_name, std::string setting, std::string value);

  // The name of the backend. Default is and empty string.
  std::string backend_name_;
  // The name of the setting. Default is and empty string.
  std::string setting_;
  // The setting value. Default is and empty string.
  std::string value_;
};

//==============================================================================
/// Server options that are used to initialize Triton Server.
///
struct ServerOptions {
  ServerOptions(std::vector<std::string> model_repository_paths);

  ServerOptions(
      std::vector<std::string> model_repository_paths, LoggingOptions logging,
      MetricsOptions metrics, std::vector<BackendConfig> be_config,
      std::string server_id, std::string backend_dir,
      std::string repo_agent_dir, bool disable_auto_complete_config,
      ModelControlMode model_control_mode);

  // Paths to model repository directory. Note that if a model is not unique
  // across all model repositories at any time, the model will not be available.
  std::vector<std::string> model_repository_paths_;
  // Logging options.
  LoggingOptions logging_;
  // Metrics options.
  MetricsOptions metrics_;
  // Backend configuration.
  std::vector<BackendConfig> be_config_;
  // The ID of the server.
  std::string server_id_;
  // The global directory searched for backend shared libraries. Default is
  // "/opt/tritonserver/backends".
  std::string backend_dir_;
  // The global directory searched for repository agent shared libraries.
  // Default is "/opt/tritonserver/repoagents".
  std::string repo_agent_dir_;
  // If set, disables the triton and backends from auto completing model
  // configuration files. Model configuration files must be provided and
  // all required configuration settings must be specified. Default is false.
  bool disable_auto_complete_config_;
  // Specify the mode for model management. Options are "MODEL_CONTROL_NONE",
  // "MODEL_CONTROL_POLL" and "MODEL_CONTROL_EXPLICIT". Default is
  // "MODEL_CONTROL_NONE".
  ModelControlMode model_control_mode_;
};

//==============================================================================
/// Structure to hold repository index for 'ModelIndex' function.
///
struct RepositoryIndex {
  RepositoryIndex(std::string name, std::string version, std::string state);

  std::string name_;     // the name of the model
  std::string version_;  // the version of the model
  std::string state_;    // the state of the model
};

//==============================================================================
/// Structure to hold buffer for output tensors.
///
struct Buffer {
  Buffer(
      const char* buffer, size_t byte_size, std::string memory_type,
      int64_t memory_type_id);

  const char* buffer_;       // the pointer to the start of the buffer
  size_t byte_size_;         // the size of buffer in bytes
  std::string memory_type_;  // the memory type of the output
  int64_t memory_type_id_;   // the memory type ID of the output
};

//==============================================================================
/// Structure to hold the inference result in the form of buffers.
///
struct BufferResult {
  // Indicates if the result has an error. If so, should not retreive the result
  // from buffer_map.
  bool has_error_;
  // The error message. Empty if no error.
  std::string error_msg_;
  /// An unordered map which the key is the name of the output tensor and the
  /// value is the Buffer object that contains the output tensor.
  std::unordered_map<std::string, Buffer> buffer_map_;
};

//==============================================================================
/// Object that encapsulates in-process C API functionalities.
///
class TritonServer {
 public:
  TritonServer(ServerOptions server_options);

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

  /// Get the set of names of models that are loaded and ready for inference.
  /// \param loaded_models Returns the set of names of models that are loaded
  /// and ready for inference
  /// \return Error object indicating success or failure.
  Error LoadedModels(std::set<std::string>* loaded_models);

  /// Get the index of model repository contents.
  /// \param repository_index Returns a vector of RepositoryIndex object
  /// representing the repository index.
  /// \return Error object indicating success or failure.
  Error ModelIndex(std::vector<RepositoryIndex>& repository_index);

  /// Get the metrics of the server.
  /// \param metrics_str Returns a string representing the metrics.
  /// \return Error object indicating success or failure.
  Error Metrics(std::string* metrics_str);

  /// Run asynchronous inference on server.
  /// \param result_future Returns the result of inference as a future of
  /// InferResult object.
  /// \param infer_request The InferRequest object contains
  /// the inputs, outputs and infer options for an inference request.
  /// \return Error object indicating success or failure.
  Error AsyncInfer(
      std::future<InferResult*>* result_future,
      const InferRequest& infer_request);

  /// Run asynchronous inference on server.
  /// \param buffer_result_future Returns the result of inference as a future of
  /// BufferResult object.
  /// \param infer_request The InferRequest object contains
  /// the inputs, outputs and infer options for an inference request.
  /// \return Error object indicating success or failure.
  Error AsyncInfer(
      std::future<BufferResult*>* buffer_result_future,
      const InferRequest& infer_request);

 private:
  Error InitializeAllocator();

  Error PrepareInferenceRequest(
      TRITONSERVER_InferenceRequest** irequest, const InferRequest& request);

  Error PrepareInferenceInput(
      TRITONSERVER_InferenceRequest* irequest, const InferRequest& request);

  Error PrepareInferenceOutput(
      TRITONSERVER_InferenceRequest* irequest, const InferRequest& request);

  Error AsyncInferHelper(
      TRITONSERVER_InferenceRequest** irequest,
      const InferRequest& infer_request);

  // Helper function for parsing data type and shape of an input tensor from
  // model configuration when 'data_type' or 'shape' field is missing.
  Error ParseDataTypeAndShape(
      const std::string model_name, const int64_t model_version,
      const std::string input_name, TRITONSERVER_DataType* datatype,
      std::vector<int64_t>* shape);

  // The server object.
  TRITONSERVER_Server* server_;
  // The allocator object allocating output tensor.
  TRITONSERVER_ResponseAllocator* allocator_;
};

//==============================================================================
/// Structure to hold options for Inference Request.
///
struct InferOptions {
  InferOptions();

  InferOptions(const std::string& model_name);

  InferOptions(
      const std::string& model_name, const int64_t model_version,
      const std::string request_id, const uint64_t correlation_id,
      const std::string correlation_id_str, const bool sequence_start,
      const bool sequence_end, const uint64_t priority,
      const uint64_t request_timeout, Allocator* custom_allocator);

  /// The name of the model to run inference.
  std::string model_name_;
  /// The version of the model to use while running inference. The default
  /// value is "-1" which means the server will select the
  /// version of the model based on its internal policy.
  int64_t model_version_;
  /// An identifier for the request. If specified will be returned
  /// in the response. Default value is an empty string which means no
  /// request_id will be used.
  std::string request_id_;
  /// The correlation ID of the inference request to be an unsigned integer.
  /// Default is 0, which indicates that the request has no correlation ID.
  uint64_t correlation_id_;
  /// The correlation ID of the inference request to be a string.
  /// Default value is "".
  std::string correlation_id_str_;
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

  /// User-provided custom reponse allocator object. Default is nullptr.
  Allocator* custom_allocator_;
};

//==============================================================================
/// Object that describes an inflight inference request.
///
class InferRequest {
 public:
  InferRequest(InferOptions infer_options);

  /// Add an input tensor to be sent within an InferRequest object.
  /// \param name The name of the input tensor.
  /// \param buffer_ptr The buffer pointer of the input data.
  /// \param byte_size The size, in bytes, of the input data.
  /// \param data_type The data type of the input. This field is optional.
  /// \param shape The shape of the input. This field is optional.
  /// \param memory_type The memory type of the input.
  /// This field is optional. Default is CPU.
  /// \param memory_type_id The memory type id of the input.
  /// This field is optional. Default is 0.
  /// \return Error object indicating success or failure.
  Error AddInput(
      const std::string name, char* buffer_ptr, const uint64_t byte_size,
      std::string data_type = "", std::vector<int64_t> shape = {},
      const MemoryType memory_type = CPU, const int64_t memory_type_id = 0);

  /// Add an input tensor to be sent within an InferRequest object.
  /// \param model_name The name of the input tensor.
  /// \param begin The begin iterator of the container.
  /// \param end  The end iterator of the container.
  /// \param data_type The data type of the input. This field is optional.
  /// \param shape The shape of the input. This field is optional.
  /// \param memory_type TThe memory type of the input.
  /// This field is optional. Default is CPU.
  /// \param memory_type_id The memory type id of the input.
  /// This field is optional. Default is 0.
  /// \return Error object indicating success or failure.
  template <typename Iterator>
  Error AddInput(
      const std::string name, Iterator& begin, Iterator& end,
      std::string data_type = "", std::vector<int64_t> shape = {},
      const MemoryType memory_type = CPU, const int64_t memory_type_id = 0);

  /// Add an requested output tensor to be sent within an InferRequest object.
  /// \param name The name of the requested output tensor.
  /// \return Error object indicating success or failure.
  Error AddRequestedOutputName(const std::string name);

  /// Clear inputs and outputs of the request except for the callback functions.
  /// This allows users to reuse the InferRequest object if needed.
  /// \return Error object indicating success or failure.
  Error Reset();

  friend class TritonServer;

 private:
  InferOptions infer_options_;
  std::list<std::string> str_bufs_;
  std::vector<InferInput*> inputs_ = {};
  std::vector<InferRequestedOutput*> outputs_ = {};
};

//==============================================================================
/// An interface for InferResult object to interpret the response to an
/// inference request.
///
class InferResult {
 public:
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
  /// \param datatype Returns the datatype of result for specified output name.
  /// \return Error object indicating success or failure.
  Error DataType(const std::string& output_name, std::string* datatype);

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
      const std::string output_name, const char** buf, size_t* byte_size);

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
  Error DebugString(std::string* string_result);

  /// Returns if there is an error within this result. If so, should not call
  /// other member functions to retreive result. \return True if this
  /// InferResult object has an error, false if no error.
  bool HasError();

  /// Returns the error message of the error.
  /// \return The messsage for the error. Empty if no error.
  std::string ErrorMsg();

 protected:
  const char* model_name_;
  int64_t model_version_;
  const char* request_id_;
  std::vector<ResponseParameters*> params_;
  std::unordered_map<std::string, InferOutput*> infer_outputs_;
  Error response_error_;
};

//==============================================================================
/// Clear all the responses that have been completed. Calling this
/// function means buffers for all the output tensors associated with the
/// responses are freed, so must not access those buffers after calling this
/// function.
void ClearCompletedResponses();

//==============================================================================
/// Custom Allocator object for providing custom functions for allocator.
/// If there are no custom functions set, will use the provided default
/// functions for allocator.
///
class Allocator {
  /***
  * ResponseAllocatorAllocFn_t: The custom response allocation for the model. If
  not set, will use the provided default allocator.

  * ResponseAllocatorReleaseFn_t: The custom response release callback function
  for the model. If not set, will use the provided default response release
  callback function.

  * ResponseAllocatorStartFn_t: The custom start callback function for the
  model. If not set, will not provide any start callback function as it’s
  typically not used.

  The signature of each function:
   * typedef Error (*ResponseAllocatorAllocFn_t)(
      const char* tensor_name, size_t byte_size, MemoryType
  preferred_memory_type, int64_t preferred_memory_type_id, void* userp, void**
  buffer, void** buffer_userp, MemoryType* actual_memory_type, int64_t*
  actual_memory_type_id);

   * typedef Error (*ResponseAllocatorReleaseFn_t)(
      void* buffer, void* buffer_userp, size_t byte_size, MemoryType
  memory_type, int64_t memory_type_id);

   * typedef Error (*ResponseAllocatorStartFn_t)(void* userp);
  ***/
 public:
  explicit Allocator(
      ResponseAllocatorAllocFn_t alloc_fn,
      ResponseAllocatorReleaseFn_t release_fn,
      ResponseAllocatorStartFn_t start_fn = nullptr)
      : alloc_fn_(alloc_fn), release_fn_(release_fn), start_fn_(start_fn)
  {
  }

  /// The functions or objects below should not be called or accessed in one's
  /// application.
  //==============================================================================
  ResponseAllocatorAllocFn_t AllocFn() { return alloc_fn_; }
  ResponseAllocatorReleaseFn_t ReleaseFn() { return release_fn_; }
  ResponseAllocatorStartFn_t StartFn() { return start_fn_; }

 private:
  ResponseAllocatorAllocFn_t alloc_fn_;
  ResponseAllocatorReleaseFn_t release_fn_;
  ResponseAllocatorStartFn_t start_fn_;
  //==============================================================================
};

}}}  // namespace triton::server::wrapper
