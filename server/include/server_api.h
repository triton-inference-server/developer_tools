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
#include "common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace triton_developer_tools { namespace server {

class InferResult;
class InferRequest;
class Allocator;

//==============================================================================
/// Structure to hold logging options for server parameters.
///
struct logging_options {
  logging_options()
      : verbose(false), info(true), warn(true), error(true),
        format(LOG_DEFAULT), log_file("")
  {
  }
  logging_options(
      bool verbose, bool info, bool warn, bool error, LogFormat format,
      std::string log_file)
      : verbose(verbose), info(info), warn(warn), error(error), format(format),
        log_file(log_file)
  {
  }


  bool verbose;          // enable/disable verbose logging level
  bool info;             // enable/disable info logging level
  bool warn;             // enable/disable warn logging level
  bool error;            // enable/disable error logging level
  LogFormat format;      // logging format
  std::string log_file;  // logging output file
};

//==============================================================================
/// Structure to hold metrics options for server parameters.
///
struct metrics_options {
  metrics_options()
      : allow_metrics(true), allow_gpu_metrics(true), metrics_interval_ms(2000)
  {
  }
  metrics_options(
      bool allow_metrics, bool allow_gpu_metrics, uint64_t metrics_interval_ms)
      : allow_metrics(allow_metrics), allow_gpu_metrics(allow_gpu_metrics),
        metrics_interval_ms(metrics_interval_ms)
  {
  }

  bool allow_metrics;            // enable/disable metrics
  bool allow_gpu_metrics;        // enable/disable GPU metrics
  uint64_t metrics_interval_ms;  // the interval for metrics collection
};

//==============================================================================
/// Structure to hold backend configuration for server parameters.
///
struct backend_config {
  backend_config() : backend_name(""), setting(""), value("") {}
  backend_config(
      std::string backend_name, std::string setting, std::string value)
      : backend_name(backend_name), setting(setting), value(value)
  {
  }

  std::string backend_name;  // the name of the backend
  std::string setting;       // the name of the setting
  std::string value;         // the setting value
};

//==============================================================================
/// Server parameters that are used to initialize Triton Server.
///
struct ServerParams {
  explicit ServerParams(std::vector<std::string> model_repository_paths)
      : model_repository_paths(model_repository_paths)
  {
    logging = logging_options();
    metrics = metrics_options();
    be_config.clear();
    server_id = "triton";
    backend_dir = "/opt/tritonserver/backends";
    repo_agent_dir = "/opt/tritonserver/repoagents";
    disable_auto_complete_config = false;
    model_control_mode = MODEL_CONTROL_NONE;
  }

  explicit ServerParams(
      std::vector<std::string> model_repository_paths, logging_options logging,
      metrics_options metrics, std::vector<backend_config> be_config,
      std::string server_id, std::string backend_dir,
      std::string repo_agent_dir, bool disable_auto_complete_config,
      ModelControlMode model_control_mode)
      : model_repository_paths(model_repository_paths), logging(logging),
        metrics(metrics), be_config(be_config), server_id(server_id),
        backend_dir(backend_dir), repo_agent_dir(repo_agent_dir),
        disable_auto_complete_config(disable_auto_complete_config),
        model_control_mode(model_control_mode)
  {
  }

  std::vector<std::string>
      model_repository_paths;  // directory paths of model repositories
  logging_options logging;     // logging options
  metrics_options metrics;     // metrics options
  std::vector<backend_config> be_config;  // backend configuration
  std::string server_id;                  // the ID of the server
  std::string backend_dir;                // directory path of backend
  std::string repo_agent_dir;             // directory path of repo agent
  bool disable_auto_complete_config;      // enable/disable auto-complete model
                                          // configuration
  ModelControlMode model_control_mode;    // model control mode
};

//==============================================================================
/// Structure to hold repository index for 'ModelIndex' function.
///
struct RepositoryIndex {
  explicit RepositoryIndex(
      std::string name, std::string version, std::string state)
      : name(name), version(version), state(state)
  {
  }

  std::string name;     // the name of the model
  std::string version;  // the version of the model
  std::string state;    // the state of the model
};

//==============================================================================
/// Structure to hold buffer for output tensors for 'AsyncInfer' function.
///
struct Buffer {
  explicit Buffer(const char* buffer, size_t byte_size)
      : buffer(buffer), byte_size(byte_size)
  {
  }

  const char* buffer;  // the pointer to the start of the buffer.
  size_t byte_size;    // the size of buffer in bytes.
};

//==============================================================================
/// Object that encapsulates in-process C API functionalities.
///
class TritonServer {
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
  /// \param infer_result Returns the result of inference as InferResult object.
  /// \param infer_request The InferRequest object contains the inputs, outputs
  /// and infer options for an inference request.
  /// \return Error object indicating success or failure.
  Error AsyncInfer(InferResult* infer_result, InferRequest infer_request);

  /// Run asynchronous inference on server.
  /// \param buffer_map Returns the result of inference as an unordered map
  /// which the key is the name of the output tensor and the value is the Buffer
  /// object that contains the output tensor.
  /// \param infer_request The InferRequest object contains the inputs, outputs
  /// and infer options for an inference request.
  /// \return Error object indicating success or failure.
  Error AsyncInfer(
      std::unordered_map<std::string, Buffer>* buffer_map,
      InferRequest infer_request);

  /// Clear all the responses that the server completed. Calling this
  /// function means the buffers for output tensors associated with the
  /// responses are freed, so must not access those buffers after calling this
  /// function.
  void ClearCompletedResponses();

  /// The functions or objects below should not be called or accessed in one's
  /// application.
  //==============================================================================
 private:
  Error InitializeAllocator(InferRequest* request);

  Error PrepareInferenceRequest(
      TRITONSERVER_InferenceRequest** irequest, InferRequest* request);

  Error PrepareInferenceInput(
      TRITONSERVER_InferenceRequest* irequest, InferRequest* request);

  Error PrepareInferenceOutput(
      TRITONSERVER_InferenceRequest* irequest, InferRequest* request);

  Error AsyncExecute(
      TRITONSERVER_InferenceRequest* irequest,
      std::future<TRITONSERVER_InferenceResponse*>* future);

  Error FinalizeResponse(
      InferResult* infer_result,
      std::future<TRITONSERVER_InferenceResponse*> future);

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
  // The responses that server completed. These reponses will not be deleted
  // until function 'ClearCompletedResponses' or destructor is called so that
  // the output buffer can still be accessed by users after inference is
  // completed.
  std::vector<TRITONSERVER_InferenceResponse*> completed_responses_;
  //==============================================================================
};

//==============================================================================
/// Structure to hold options for Inference Request.
///
struct InferOptions {
  explicit InferOptions(const std::string& model_name)
      : model_name(model_name), model_version(-1), request_id(""),
        correlation_id(0), correlation_id_str(""), sequence_start(false),
        sequence_end(false), priority(0), request_timeout(0),
        custom_allocator(nullptr)
  {
  }

  explicit InferOptions(
      const std::string& model_name, const int64_t model_version,
      const std::string request_id, const uint64_t correlation_id,
      const std::string correlation_id_str, const bool sequence_start,
      const bool sequence_end, const uint64_t priority,
      const uint64_t request_timeout, Allocator* custom_allocator)
      : model_name(model_name), model_version(model_version),
        request_id(request_id), correlation_id(correlation_id),
        correlation_id_str(correlation_id_str), sequence_start(sequence_start),
        sequence_end(sequence_end), priority(priority),
        request_timeout(request_timeout), custom_allocator(custom_allocator)
  {
  }

  /// The name of the model to run inference.
  std::string model_name;
  /// The version of the model to use while running inference. The default
  /// value is an empty string which means the server will select the
  /// version of the model based on its internal policy.
  int64_t model_version;
  /// An identifier for the request. If specified will be returned
  /// in the response. Default value is an empty string which means no
  /// request_id will be used.
  std::string request_id;
  /// The correlation ID of the inference request to be an unsigned integer.
  /// Default is 0, which indicates that the request has no correlation ID.
  uint64_t correlation_id;
  /// The correlation ID of the inference request to be a string.
  /// Default value is "".
  std::string correlation_id_str;
  /// Indicates whether the request being added marks the start of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_start;
  /// Indicates whether the request being added marks the end of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_end;
  /// Indicates the priority of the request. Priority value zero
  /// indicates that the default priority level should be used
  /// (i.e. same behavior as not specifying the priority parameter).
  /// Lower value priorities indicate higher priority levels. Thus
  /// the highest priority level is indicated by setting the parameter
  /// to 1, the next highest is 2, etc. If not provided, the server
  /// will handle the request using default setting for the model.
  uint64_t priority;
  /// The timeout value for the request, in microseconds. If the request
  /// cannot be completed within the time by the server can take a
  /// model-specific action such as terminating the request. If not
  /// provided, the server will handle the request using default setting
  /// for the model.
  uint64_t request_timeout;

  /// User-provided custom reponse allocator object.
  Allocator* custom_allocator;
};

//==============================================================================
/// Object that describes an inflight inference request.
///
class InferRequest {
 public:
  InferRequest(InferOptions infer_options);

  ~InferRequest(){};

  /// Add an input tensor to be sent within an InferRequest object.
  /// \param name The name of the input tensor.
  /// \param buffer_ptr The buffer pointer of the input data.
  /// \param byte_size The size, in bytes, of the input data.
  /// \param data_type The type of the input. This field is optional.
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
  /// \param data_type The data type of the input tensor. This is optional.
  /// \param shape The shape of the input tensor. This is optional.
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

  /// The functions or objects below should not be called or accessed in one's
  /// application.
  //==============================================================================
  std::string ModelName() const { return model_name_; }
  int64_t ModelVersion() const { return model_version_; }
  std::string RequestId() const { return request_id_; }
  uint64_t CorrelationId() const { return correlation_id_; };
  std::string CorrelationIdStr() const { return correlation_id_str_; }
  bool SequenceStart() const { return sequence_start_; }
  bool SequenceEnd() const { return sequence_end_; }
  uint64_t Priority() const { return priority_; }
  uint64_t RequestTimeout() const { return request_timeout_; }
  const std::vector<InferInput*> Inputs() const { return inputs_; }
  const std::vector<InferRequestedOutput*> Outputs() const { return outputs_; }

  static TRITONSERVER_Error* CustomAllocationFn(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id);
  static TRITONSERVER_Error* CustomReleaseFn(
      TRITONSERVER_ResponseAllocator* allocator, void* buffer,
      void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);
  static TRITONSERVER_Error* CustomStartFn(
      TRITONSERVER_ResponseAllocator* allocator, void* userp);

 private:
  std::string model_name_;
  int64_t model_version_;
  std::string request_id_;
  uint64_t correlation_id_;
  std::string correlation_id_str_;
  bool sequence_start_;
  bool sequence_end_;
  uint64_t priority_;
  uint64_t request_timeout_;

  std::list<std::string> str_bufs_;

  std::vector<InferInput*> inputs_ = {};
  std::vector<InferRequestedOutput*> outputs_ = {};
  //==============================================================================
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

  /// The functions or objects below should not be called or accessed in one's
  /// application.
  //==============================================================================
  Error SetResultInfo(
      const char* model_name, int64_t model_version, const char* request_id);

  void AddInferOutput(const std::string name, InferOutput* output)
  {
    infer_outputs_[name] = output;
  }

  std::unordered_map<std::string, InferOutput*> Outputs()
  {
    return infer_outputs_;
  }

  std::vector<ResponseParameters*> Params() { return params_; }

 private:
  const char* model_name_;
  int64_t model_version_;
  const char* request_id_;
  std::vector<ResponseParameters*> params_;
  std::unordered_map<std::string, InferOutput*> infer_outputs_;
  //==============================================================================
};

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

}}}  // namespace triton::triton_developer_tools::server
