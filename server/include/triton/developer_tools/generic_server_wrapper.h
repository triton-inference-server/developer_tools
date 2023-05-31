// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>
#include "../src/infer_requested_output.h"
#include "../src/tracer.h"
#include "common.h"

namespace triton { namespace developer_tools { namespace server {

class ServerOptions;
class InferOptions;
class RepositoryIndex;
class NewModelRepo;
class GenericInferRequest;
class GenericInferResult;
class Tensor;
class GenericInferRequest;
class GenericInferResult;
using TensorAllocMap = std::unordered_map<
    std::string,
    std::tuple<const void*, size_t, TRITONSERVER_MemoryType, int64_t>>;

//==============================================================================
/// Object that encapsulates in-process C API functionalities.
///
class GenericTritonServer {
 public:
  ///  Create a GenericTritonServer instance.
  static std::unique_ptr<GenericTritonServer> Create(
      const ServerOptions& server_options);

  virtual ~GenericTritonServer();

  /// Load the requested model or reload the model if it is already loaded.
  /// \param model_name The name of the model.
  virtual void LoadModel(const std::string& model_name) = 0;

  /// Unload the requested model. Unloading a model that is not loaded
  /// on server has no affect.
  /// \param model_name The name of the model.
  virtual void UnloadModel(const std::string& model_name) = 0;

  /// Get the set of names of models that are loaded and ready for inference.
  /// \return Returns the set of names of models that are
  /// loaded and ready for inference.
  virtual std::set<std::string> LoadedModels() = 0;

  /// Get the index of model repository contents.
  /// \return Returns a vector of 'RepositoryIndex' object
  /// representing the repository index.
  virtual std::vector<RepositoryIndex> ModelIndex() = 0;

  /// Get the metrics of the server.
  /// \return Returns a string representing the metrics.
  virtual std::string ServerMetrics() = 0;

  /// Get the inference statistics of the specified model.
  /// \param model_name The name of the model.
  /// \param model_version the version of the model requested.
  /// \return Returns a json string representing the model metrics.
  virtual std::string ModelStatistics(
      const std::string& model_name, const int64_t model_version) = 0;

  /// Is the server live?
  /// \return Returns true if server is live, false otherwise.
  virtual bool IsServerLive() = 0;

  /// Is the server ready?
  /// \return Returns true if server is ready, false otherwise.
  virtual bool IsServerReady() = 0;

  /// Stop a server object. A server can't be restarted once it is
  /// stopped.
  virtual void ServerStop() = 0;

  /// Is the model ready?
  /// \param model_name The name of the model to get readiness for.
  /// \param model_version The version of the model to get readiness
  /// for.  If -1 then the server will choose a version based on the
  /// model's policy. This field is optional, default is -1.
  /// \return Returns true if server is ready, false otherwise.
  virtual bool IsModelReady(
      const std::string& model_name, const int64_t model_version = -1) = 0;

  /// Get the configuration of specified model.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model to get configuration.
  /// The default value is -1 which means then the server will
  /// choose a version based on the model and internal policy. This field is
  /// optional. \return Returns JSON representation of model configuration as a
  /// string.
  virtual std::string ModelConfig(
      const std::string& model_name, const int64_t model_version = -1) = 0;

  /// Get the metadata of the server.
  /// \return Returns JSON representation of server metadata as a string.
  virtual std::string ServerMetadata() = 0;

  /// Get the metadata of specified model.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model to get configuration.
  /// The default value is -1 which means then the server will choose a version
  /// based on the model and internal policy. This field is optional.
  /// \return Returns JSON representation of model metadata as a string.
  virtual std::string ModelMetadata(
      const std::string& model_name, const int64_t model_version = -1) = 0;

  /// Register a new model repository. This function is not available in polling
  /// mode.
  /// \param new_model_repo The 'NewModelRepo' object contains the info of the
  /// new model repo to be registered.
  virtual void RegisterModelRepo(const NewModelRepo& new_model_repo) = 0;

  /// Unregister a model repository. This function is not available in polling
  /// mode.
  /// \param repo_path The full path to the model repository.
  virtual void UnregisterModelRepo(const std::string& repo_path) = 0;

  virtual std::unique_ptr<GenericInferResult> Infer(
      GenericInferRequest& infer_request) = 0;
};

//==============================================================================
/// An interface for InferResult object to interpret the response to an
/// inference request.
///
class GenericInferResult {
 public:
  virtual ~GenericInferResult();

  /// Get the name of the model which generated this response.
  /// \return Returns the name of the model.
  virtual std::string ModelName() noexcept = 0;

  /// Get the version of the model which generated this response.
  /// \return Returns the version of the model.
  virtual std::string ModelVersion() noexcept = 0;

  /// Get the id of the request which generated this response.
  /// \return Returns the id of the request.
  virtual std::string Id() noexcept = 0;

  /// Get the output names from the infer result
  /// \return Vector of output names
  virtual std::vector<std::string> OutputNames() = 0;

  /// Get the result output as a shared pointer of 'Tensor' object. The 'buffer'
  /// field of the output is owned by the returned 'Tensor' object itself. Note
  /// that for string data, need to use 'StringData' function for string data
  /// result.
  /// \param name The name of the output tensor to be retrieved.
  /// \return Returns the output result as a shared pointer of 'Tensor' object.
  virtual std::shared_ptr<Tensor> Output(const std::string& name) = 0;

  /// Get the result data as a vector of strings. The vector will
  /// receive a copy of result data. An exception will be thrown if
  /// the data type of output is not 'BYTES'.
  /// \param output_name The name of the output to get result data.
  /// \return Returns the result data represented as a vector of strings. The
  /// strings are stored in the row-major order.
  virtual std::vector<std::string> StringData(
      const std::string& output_name) = 0;

  /// Return the complete response as a user friendly string.
  /// \return The string describing the complete response.
  virtual std::string DebugString() = 0;

  /// Return if there is an error within this result.
  /// \return True if this 'GenericInferResult' object has an error, false if no
  /// error.
  virtual bool HasError() = 0;

  /// Return the error message of the error.
  /// \return The messsage for the error. Empty if no error.
  virtual std::string ErrorMsg() = 0;
};

//==============================================================================
/// Object that describes an inflight inference request.
///
class GenericInferRequest {
 public:
  ///  Create an InferRequest instance.
  static std::unique_ptr<GenericInferRequest> Create(
      const InferOptions& infer_options);

  virtual ~GenericInferRequest();

  /// Add an input tensor to be sent within an InferRequest object. The input
  /// data buffer within the 'Tensor' object must not be modified until
  /// inference is completed and result is returned.
  /// \param name The name of the input tensor.
  /// \param input A Tensor object that describes an input tensor.
  virtual void AddInput(
      const std::string& name, const Tensor& input) noexcept = 0;

  /// Add a requested output to be sent within an InferRequest object.
  /// Calling this function is optional. If no output(s) are specifically
  /// requested then all outputs defined by the model will be calculated and
  /// returned. Pre-allocated buffer for each output should be specified within
  /// the 'Tensor' object.
  /// \param name The name of the output tensor.
  /// \param output A Tensor object that describes an output tensor containing
  /// its pre-allocated buffer.
  virtual void AddRequestedOutput(const std::string& name, Tensor& output) = 0;

  /// Add a requested output to be sent within an InferRequest object.
  /// Calling this function is optional. If no output(s) are specifically
  /// requested then all outputs defined by the model will be calculated and
  /// returned.
  /// \param name The name of the output tensor.
  virtual void AddRequestedOutput(const std::string& name) = 0;

  /// Clear inputs and outputs of the request. This allows users to reuse the
  /// InferRequest object if needed.
  virtual void Reset() = 0;
};

}}}  // namespace triton::developer_tools::server
