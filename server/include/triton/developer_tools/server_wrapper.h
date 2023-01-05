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

#include <climits>
#include <future>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include "generic_server_wrapper.h"
#include "triton/core/tritonserver.h"
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace developer_tools { namespace server {

class Allocator;
class InferResult;
class InferRequest;
struct ResponseParameters;
class TraceManager;

//==============================================================================
/// Object that encapsulates in-process C API functionalities.
///
class TritonServer : public GenericTritonServer {
 public:
  static std::unique_ptr<TritonServer> Create(
      const ServerOptions& server_options);

  virtual ~TritonServer();

  /// Load the requested model or reload the model if it is already loaded.
  /// \param model_name The name of the model.
  void LoadModel(const std::string& model_name) override;

  /// Unload the requested model. Unloading a model that is not loaded
  /// on server has no affect.
  /// \param model_name The name of the model.
  void UnloadModel(const std::string& model_name) override;

  /// Get the set of names of models that are loaded and ready for inference.
  /// \return Returns the set of names of models that are
  /// loaded and ready for inference.
  std::set<std::string> LoadedModels() override;

  /// Get the index of model repository contents.
  /// \return Returns a vector of 'RepositoryIndex' object
  /// representing the repository index.
  std::vector<RepositoryIndex> ModelIndex() override;

  /// Get the metrics of the server.
  /// \return Returns a string representing the metrics.
  std::string ServerMetrics() override;

  /// Get the inference statistics of the specified model.
  /// \param model_name The name of the model.
  /// \param model_version the version of the model requested.
  /// \return Returns a json string representing the model metrics.
  std::string ModelStatistics(
      const std::string& model_name, const int64_t model_version) override;

  /// Run asynchronous inference on server.
  /// \param infer_request The InferRequest object contains
  /// the inputs, outputs and infer options for an inference request.
  /// \return Returns the result of inference as a future of
  /// a unique pointer of InferResult object.
  virtual std::future<std::unique_ptr<InferResult>> AsyncInfer(
      InferRequest& infer_request) = 0;

  /// Is the server live?
  /// \return Returns true if server is live, false otherwise.
  bool IsServerLive() override;

  /// Is the server ready?
  /// \return Returns true if server is ready, false otherwise.
  bool IsServerReady() override;

  /// Stop a server object. A server can't be restarted once it is
  /// stopped.
  void ServerStop() override;

  /// Is the model ready?
  /// \param model_name The name of the model to get readiness for.
  /// \param model_version The version of the model to get readiness
  /// for.  If -1 then the server will choose a version based on the
  /// model's policy. This field is optional, default is -1.
  /// \return Returns true if server is ready, false otherwise.
  bool IsModelReady(
      const std::string& model_name, const int64_t model_version = -1) override;

  /// Get the configuration of specified model.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model to get configuration.
  /// The default value is -1 which means then the server will
  /// choose a version based on the model and internal policy. This field is
  /// optional. \return Returns JSON representation of model configuration as a
  /// string.
  std::string ModelConfig(
      const std::string& model_name, const int64_t model_version = -1) override;

  /// Get the metadata of the server.
  /// \return Returns JSON representation of server metadata as a string.
  std::string ServerMetadata() override;

  /// Get the metadata of specified model.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model to get configuration.
  /// The default value is -1 which means then the server will choose a version
  /// based on the model and internal policy. This field is optional.
  /// \return Returns JSON representation of model metadata as a string.
  std::string ModelMetadata(
      const std::string& model_name, const int64_t model_version = -1) override;

  /// Register a new model repository. This function is not available in polling
  /// mode.
  /// \param new_model_repo The 'NewModelRepo' object contains the info of the
  /// new model repo to be registered.
  void RegisterModelRepo(const NewModelRepo& new_model_repo) override;

  /// Unregister a model repository. This function is not available in polling
  /// mode.
  /// \param repo_path The full path to the model repository.
  void UnregisterModelRepo(const std::string& repo_path) override;

 protected:
  void PrepareInferenceRequest(
      TRITONSERVER_InferenceRequest** irequest, const InferRequest& request);

  void PrepareInferenceInput(
      TRITONSERVER_InferenceRequest* irequest, const InferRequest& request);

  void PrepareInferenceOutput(
      TRITONSERVER_InferenceRequest* irequest, InferRequest& request);

  void AsyncInferHelper(
      TRITONSERVER_InferenceRequest** irequest,
      const InferRequest& infer_request);

  // The server object.
  std::shared_ptr<TRITONSERVER_Server> server_;
  // The allocator object allocating output tensor.
  TRITONSERVER_ResponseAllocator* allocator_;
  // The trace manager.
  std::shared_ptr<TraceManager> trace_manager_;
};


//==============================================================================
/// An interface for InferResult object to interpret the response to an
/// inference request.
///
class InferResult : public GenericInferResult {
 public:
  virtual ~InferResult();

  /// Get the name of the model which generated this response.
  /// \return Returns the name of the model.
  std::string ModelName() noexcept override;

  /// Get the version of the model which generated this response.
  /// \return Returns the version of the model.
  std::string ModelVersion() noexcept override;

  /// Get the id of the request which generated this response.
  /// \return Returns the id of the request.
  std::string Id() noexcept override;

  /// Get the output names from the infer result
  /// \return Vector of output names
  std::vector<std::string> OutputNames() override;
  /// Get the result output as a shared pointer of 'Tensor' object. The 'buffer'
  /// field of the output is owned by the returned 'Tensor' object itself. Note
  /// that for string data, need to use 'StringData' function for string data
  /// result.
  /// \param name The name of the output tensor to be retrieved.
  /// \return Returns the output result as a shared pointer of 'Tensor' object.
  std::shared_ptr<Tensor> Output(const std::string& name) override;

  /// Get the result data as a vector of strings. The vector will
  /// receive a copy of result data. An exception will be thrown if
  /// the data type of output is not 'BYTES'.
  /// \param output_name The name of the output to get result data.
  /// \return Returns the result data represented as a vector of strings. The
  /// strings are stored in the row-major order.
  std::vector<std::string> StringData(const std::string& output_name) override;

  /// Return the complete response as a user friendly string.
  /// \return The string describing the complete response.
  std::string DebugString() override;

  /// Return if there is an error within this result.
  /// \return True if this 'InferResult' object has an error, false if no error.
  bool HasError() override;

  /// Return the error message of the error.
  /// \return The messsage for the error. Empty if no error.
  std::string ErrorMsg() override;

  // Get the pointer to the future of the next result. This function is used for
  // retrieving multiple responses from decoupled model. If there is no next
  // result, this function will return nullptr.
  std::unique_ptr<std::future<std::unique_ptr<InferResult>>> GetNextResult();

  friend class InternalServer;

 protected:
  InferResult();
  const char* model_name_;
  int64_t model_version_;
  const char* request_id_;
  std::vector<std::unique_ptr<ResponseParameters>> params_;
  std::unordered_map<std::string, std::shared_ptr<Tensor>> infer_outputs_;
  bool has_error_;
  std::string error_msg_;

  // The pointer to the future of the next result.
  std::unique_ptr<std::future<std::unique_ptr<InferResult>>>
      next_result_future_;

  TRITONSERVER_InferenceResponse* completed_response_;
};

//==============================================================================
/// Object that describes an inflight inference request.
///
class InferRequest : public GenericInferRequest {
 public:
  ///  Create an InferRequest instance.
  static std::unique_ptr<InferRequest> Create(
      const InferOptions& infer_options);

  ~InferRequest();

  /// Add an input tensor to be sent within an InferRequest object. The input
  /// data buffer within the 'Tensor' object must not be modified until
  /// inference is completed and result is returned.
  /// \param name The name of the input tensor.
  /// \param input A Tensor object that describes an input tensor.
  void AddInput(const std::string& name, const Tensor& input) noexcept override;

  /// Add an input tensor to be sent within an InferRequest object. This
  /// function is for containers holding 'non-string' data elements. Data in the
  /// container should be contiguous, and the the container must not be modified
  /// until inference is completed and result is returned.
  /// \param name The name of the input tensor.
  /// \param begin The begin iterator of the container.
  /// \param end  The end iterator of the container.
  /// \param data_type The data type of the input.
  /// \param shape The shape of the input.
  /// \param memory_type The memory type of the input.
  /// \param memory_type_id The ID of the memory for the tensor. (e.g. '0' is
  /// the memory type id of 'GPU-0')
  template <
      typename Iterator,
      typename std::enable_if<std::is_same<
          typename std::iterator_traits<Iterator>::value_type,
          std::string>::value>::type* = nullptr>
  void AddInput(
      const std::string& name, const Iterator begin, const Iterator end,
      const DataType& data_type, const std::vector<int64_t>& shape,
      const MemoryType& memory_type, const int64_t memory_type_id) noexcept;

  /// Add an input tensor to be sent within an InferRequest object. This
  /// function is for containers holding 'string' elements. Data in the
  /// container should be contiguous, and the the container must not be modified
  /// until inference is completed and the result is returned.
  /// \param name The name of the input tensor.
  /// \param begin The begin iterator of the container.
  /// \param end  The end iterator of the container.
  /// \param data_type The data type of the input. For 'string' input, data type
  /// should be 'BYTES'.
  /// \param shape The shape of the input.
  /// \param memory_type The memory type of the input.
  /// \param memory_type_id The ID of the memory for the tensor. (e.g. '0' is
  /// the memory type id of 'GPU-0')
  template <
      typename Iterator,
      typename std::enable_if<!std::is_same<
          typename std::iterator_traits<Iterator>::value_type,
          std::string>::value>::type* = nullptr>
  void AddInput(
      const std::string& name, const Iterator begin, const Iterator end,
      const DataType& data_type, const std::vector<int64_t>& shape,
      const MemoryType& memory_type, const int64_t memory_type_id) noexcept;

  /// Add a requested output to be sent within an InferRequest object.
  /// Calling this function is optional. If no output(s) are specifically
  /// requested then all outputs defined by the model will be calculated and
  /// returned. Pre-allocated buffer for each output should be specified within
  /// the 'Tensor' object.
  /// \param name The name of the output tensor.
  /// \param output A Tensor object that describes an output tensor containing
  /// its pre-allocated buffer.
  void AddRequestedOutput(const std::string& name, Tensor& output) override;

  /// Add a requested output to be sent within an InferRequest object.
  /// Calling this function is optional. If no output(s) are specifically
  /// requested then all outputs defined by the model will be calculated and
  /// returned.
  /// \param name The name of the output tensor.
  void AddRequestedOutput(const std::string& name) override;

  /// Clear inputs and outputs of the request. This allows users to reuse the
  /// InferRequest object if needed.
  void Reset() override;
 friend class TritonServer;
 friend class InternalServer;

 protected:
  InferRequest();

  std::unique_ptr<InferOptions> infer_options_;
  std::list<std::string> str_bufs_;
  std::unordered_map<std::string, std::unique_ptr<Tensor>> inputs_;
  std::vector<std::unique_ptr<InferRequestedOutput>> outputs_;

  // The map for each output tensor and a tuple of it's pre-allocated buffer,
  // byte size, memory type and memory type id.
  TensorAllocMap tensor_alloc_map_;
  // The updated trace setting for the specified model set within
  // 'InferOptions'. If set, the lifetime of this 'TraceManager::Trace' object
  // should be long enough until the trace associated with this request is
  // written to file.
  std::shared_ptr<TraceManager::Trace> trace_;

  // If the requested model is a decoupled model. If true, the lifetime of this
  // 'InferRequest' should be long enough until all the responses are returned
  // and retrieved.
  bool is_decoupled_;
  private:   
  // The promise object used for setting value to the result future.
  std::unique_ptr<std::promise<std::unique_ptr<InferResult>>> prev_promise_;
};
//==============================================================================
/// Helper functions to convert Wrapper enum to string.
///
std::string MemoryTypeString(const MemoryType& memory_type);
std::string DataTypeString(const DataType& data_type);
std::string ModelReadyStateString(const ModelReadyState& state);

//==============================================================================
/// Implementation of template functions
///
template <
    typename Iterator, typename std::enable_if<std::is_same<
                           typename std::iterator_traits<Iterator>::value_type,
                           std::string>::value>::type* = nullptr>
void
InferRequest::AddInput(
    const std::string& name, const Iterator begin, const Iterator end,
    const DataType& data_type, const std::vector<int64_t>& shape,
    const MemoryType& memory_type, const int64_t memory_type_id) noexcept
{
  // Serialize the strings into a "raw" buffer. The first 4-bytes are
  // the length of the string length. Next are the actual string
  // characters. There is *not* a null-terminator on the string.
  str_bufs_.emplace_back();
  std::string& sbuf = str_bufs_.back();

  Iterator it;
  for (it = begin; it != end; it++) {
    auto len = it->size();
    sbuf.append(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
    sbuf.append(*it);
  }
  Tensor input(
      reinterpret_cast<char*>(&sbuf[0]), sbuf.size(), DataType::BYTES, shape,
      memory_type, memory_type_id);

  AddInput(name, input);
}

template <
    typename Iterator, typename std::enable_if<!std::is_same<
                           typename std::iterator_traits<Iterator>::value_type,
                           std::string>::value>::type* = nullptr>
void
InferRequest::AddInput(
    const std::string& name, const Iterator begin, const Iterator end,
    const DataType& data_type, const std::vector<int64_t>& shape,
    const MemoryType& memory_type, const int64_t memory_type_id) noexcept
{
  // FIXME (DLIS-4134) This function should also work for non-contiguous
  // container, and input data should be copied so that we don't need to worry
  // about the lifetime of input data.
  size_t bytes = sizeof(*begin) * std::distance(begin, end);
  Tensor input(
      reinterpret_cast<char*>(&(*begin)), bytes, data_type, shape, memory_type,
      memory_type_id);

  AddInput(name, input);
}

}}}  // namespace triton::developer_tools::server
