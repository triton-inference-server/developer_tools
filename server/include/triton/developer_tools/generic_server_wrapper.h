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
#include <memory>
#include <set>
#include <vector>
#include <unordered_map>
#include <list>
#include "common.h"
#include "../src/infer_requested_output.h"
#include "../src/tracer.h"

namespace triton { namespace developer_tools { namespace server {

class ServerOptions;
class InferOptions;
class RepositoryIndex;
class NewModelRepo;
class Tensor;
using TensorAllocMap = std::unordered_map<std::string, std::tuple<const void*, size_t, TRITONSERVER_MemoryType, int64_t>>;

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
  virtual void UnloadModel(const std::string& model_name)= 0;

  /// Get the set of names of models that are loaded and ready for inference.
  /// \return Returns the set of names of models that are
  /// loaded and ready for inference.
  virtual std::set<std::string> LoadedModels()= 0;

  /// Get the index of model repository contents.
  /// \return Returns a vector of 'RepositoryIndex' object
  /// representing the repository index.
  virtual std::vector<RepositoryIndex> ModelIndex()= 0;

  /// Get the metrics of the server.
  /// \return Returns a string representing the metrics.
  virtual std::string ServerMetrics()= 0;

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
      const std::string& model_name, const int64_t model_version = -1)= 0;

  /// Register a new model repository. This function is not available in polling
  /// mode.
  /// \param new_model_repo The 'NewModelRepo' object contains the info of the
  /// new model repo to be registered.
  virtual void RegisterModelRepo(const NewModelRepo& new_model_repo)= 0;

  /// Unregister a model repository. This function is not available in polling
  /// mode.
  /// \param repo_path The full path to the model repository.
  virtual void UnregisterModelRepo(const std::string& repo_path) = 0;
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
  virtual std::string ModelVersion() noexcept= 0;

  /// Get the id of the request which generated this response.
  /// \return Returns the id of the request.
  virtual std::string Id() noexcept= 0;

  /// Get the output names from the infer result
  /// \return Vector of output names
  virtual std::vector<std::string> OutputNames()= 0;
  /// Get the result output as a shared pointer of 'Tensor' object. The 'buffer'
  /// field of the output is owned by the returned 'Tensor' object itself. Note
  /// that for string data, need to use 'StringData' function for string data
  /// result.
  /// \param name The name of the output tensor to be retrieved.
  /// \return Returns the output result as a shared pointer of 'Tensor' object.
  virtual std::shared_ptr<Tensor> Output(const std::string& name)= 0;

  /// Get the result data as a vector of strings. The vector will
  /// receive a copy of result data. An exception will be thrown if
  /// the data type of output is not 'BYTES'.
  /// \param output_name The name of the output to get result data.
  /// \return Returns the result data represented as a vector of strings. The
  /// strings are stored in the row-major order.
  virtual std::vector<std::string> StringData(const std::string& output_name)= 0;

  /// Return the complete response as a user friendly string.
  /// \return The string describing the complete response.
  virtual std::string DebugString()= 0;

  /// Return if there is an error within this result.
  /// \return True if this 'GenericInferResult' object has an error, false if no
  /// error.
  virtual bool HasError()= 0;

  /// Return the error message of the error.
  /// \return The messsage for the error. Empty if no error.
  virtual std::string ErrorMsg()= 0;
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
  virtual void AddInput(const std::string& name, const Tensor& input) noexcept = 0;

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
//==============================================================================
/// Structure to hold logging options for setting 'ServerOptions'.
///
struct LoggingOptions {
  // The range of VerboseLevel is [0, INT_MAX].
  enum class VerboseLevel : int { OFF = 0, MIN = 1, MAX = INT_MAX };
  enum class LogFormat { DEFAULT, ISO8601 };

  LoggingOptions();

  LoggingOptions(
      const VerboseLevel verbose, const bool info, const bool warn,
      const bool error, const LogFormat& format, const std::string& log_file);

  // Verbose logging level. Default is OFF.
  VerboseLevel verbose_;
  // Enable or disable info logging level. Default is true.
  bool info_;
  // Enable or disable warn logging level. Default is true.
  bool warn_;
  // Enable or disable error logging level. Default is true.
  bool error_;
  // The format of logging. For "DEFAULT", the log severity (L) and
  // timestamp will be logged as "LMMDD hh:mm:ss.ssssss". For "ISO8601", the
  // log format will be "YYYY-MM-DDThh:mm:ssZ L". Default is 'DEFAULT'.
  LogFormat format_;
  // Logging output file. If specified, log outputs will be saved to this file.
  // If not specified, log outputs will stream to the console. Default is an
  // empty string.
  std::string log_file_;  // logging output file
};

//==============================================================================
/// Structure to hold metrics options for setting 'ServerOptions'.
/// See here for more information:
/// https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md.
struct MetricsOptions {
  MetricsOptions();

  MetricsOptions(
      const bool allow_metrics, const bool allow_gpu_metrics,
      const bool allow_cpu_metrics, const uint64_t metrics_interval_ms);

  // Enable or disable metrics. Default is true.
  bool allow_metrics_;
  // Enable or disable GPU metrics. Default is true.
  bool allow_gpu_metrics_;
  // Enable or disable CPU metrics. Default is true.
  bool allow_cpu_metrics_;
  // The interval for metrics collection. Default is 2000.
  uint64_t metrics_interval_ms_;
};

//==============================================================================
/// Structure to hold rate limit resource for setting 'ServerOptions'. See here
/// for more information:
/// https://github.com/triton-inference-server/server/blob/main/docs/user_guide/rate_limiter.md.
struct RateLimitResource {
  RateLimitResource(const std::string& name, const int count);

  RateLimitResource(const std::string& name, const int count, const int device);

  // The name of the resource.
  std::string name_;
  // The count of the resource.
  int count_;
  // The device identifier for the resource. This field is optional and if not
  // specified will be applied to every device. The device value is ignored for
  // a global resource. The server will use the rate limiter configuration
  // specified for instance groups in model config to determine whether resource
  // is global. In case of conflicting resource type in different model
  // configurations, server will raise an appropriate error while loading model.
  int device_;
};

//==============================================================================
/// Structure to hold GPU limit of model loading for setting 'ServerOptions'.
/// The limit on GPU memory usage is specified as a fraction. If model loading
/// on the device is requested and the current memory usage exceeds the limit,
/// the load will be rejected. If not specified, the limit will not be set.
struct ModelLoadGPULimit {
  ModelLoadGPULimit(const int device_id, const double& fraction);

  // The GPU device ID.
  int device_id_;
  // The limit on memory usage as a fraction.
  double fraction_;
};


//==============================================================================
/// Custom Allocator object for providing custom functions for allocator.
/// If there is no custom allocator provided, will use the default allocator.
///
class Allocator {
  /***
  * ResponseAllocatorAllocFn_t: The custom response allocation that allocates a
  buffer to hold an output tensor.

  * OutputBufferReleaseFn_t: The custom output buffer release function
  that is called to release a buffer allocated by 'ResponseAllocatorAllocFn_t'.
  This function is called in the destructor of 'Tensor' object when the output
  tensor goes out of scope. User has the responsibility to clean the buffer
  correctly.

  * ResponseAllocatorStartFn_t: The custom start callback function that is
  called to indicate that subsequent allocation requests will refer to a new
  response. If not set, will not provide any start callback function as it’s
  typically not used.

  The signature of each function:

    \param tensor_name The name of the output tensor to allocate for.
    \param byte_size The size of the buffer to allocate.
    \param memory_type The type of memory that the caller prefers for
    the buffer allocation.
    \param memory_type_id The ID of the memory that the caller prefers
    for the buffer allocation.
    \param buffer Returns a pointer to the allocated memory.
    \param actual_memory_type Returns the type of memory where the
    allocation resides. May be different than the type of memory
    requested by 'memory_type'.
    \param actual_memory_type_id Returns the ID of the memory where
    the allocation resides. May be different than the ID of the memory
    requested by 'memory_type_id'.
  * using ResponseAllocatorAllocFn_t = void (*)(const char* tensor_name,
    size_t byte_size, MemoryType memory_type, int64_t memory_type_id, void**
    buffer, MemoryType* actual_memory_type, int64_t* actual_memory_type_id);

    \param buffer Pointer to the buffer to be freed.
    \param byte_size The size of the buffer.
    \param memory_type The type of memory holding the buffer.
    \param memory_type_id The ID of the memory holding the buffer.
  * using OutputBufferReleaseFn_t = void (*)(
    void* buffer, size_t byte_size, MemoryType memory_type, int64_t
    memory_type_id);

    \param userp The user data pointer that is passed to the
    'ResponseAllocatorStartFn_t' callback function.
  * using ResponseAllocatorStartFn_t = void (*)(void* userp);
  ***/
 public:
  explicit Allocator(
      ResponseAllocatorAllocFn_t alloc_fn, OutputBufferReleaseFn_t release_fn,
      ResponseAllocatorStartFn_t start_fn = nullptr)
      : alloc_fn_(alloc_fn), release_fn_(release_fn), start_fn_(start_fn)
  {
  }

  ResponseAllocatorAllocFn_t AllocFn() { return alloc_fn_; }
  OutputBufferReleaseFn_t ReleaseFn() { return release_fn_; }
  ResponseAllocatorStartFn_t StartFn() { return start_fn_; }

 private:
  ResponseAllocatorAllocFn_t alloc_fn_;
  OutputBufferReleaseFn_t release_fn_;
  ResponseAllocatorStartFn_t start_fn_;
};

//==============================================================================
/// Structure to hold backend configuration for setting 'ServerOptions'.
/// Different Triton-supported backends have different backend configuration
/// options. Please refer to the 'Command line options' section in the
/// documentation of each backend to see the options (e.g. Tensorflow Backend:
/// https://github.com/triton-inference-server/tensorflow_backend#command-line-options)
struct BackendConfig {
  BackendConfig(
      const std::string& name, const std::string& setting,
      const std::string& value);

  // The name of the backend.
  std::string name_;
  // The name of the setting.
  std::string setting_;
  // The setting value.
  std::string value_;
};

//==============================================================================
/// Structure to hold CUDA memory pool byte size for setting 'ServerOptions'.
/// If GPU support is enabled, the server will allocate CUDA memory to minimize
/// data transfer between host and devices until it exceeds the specified byte
/// size. This will not affect the allocation conducted by the backend
/// frameworks.
struct CUDAMemoryPoolByteSize {
  CUDAMemoryPoolByteSize(const int gpu_device, const uint64_t size);

  // The GPU device ID to allocate the memory pool.
  int gpu_device_;
  // The CUDA memory pool byte size that the server can allocate on given GPU
  // device. Default is 64 MB.
  uint64_t size_;
};

//==============================================================================
/// Structure to hold host policy for setting 'ServerOptions'.
/// See here for more information:
/// https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#host-policy.
struct HostPolicy {
  enum class Setting { NUMA_NODE, CPU_CORES };

  HostPolicy(
      const std::string& name, const Setting& setting,
      const std::string& value);

  // The name of the policy.
  std::string name_;
  // The kind of the host policy setting. Currently supported settings are
  // 'NUMA_NODE', 'CPU_CORES'. Note that 'NUMA_NODE' setting will affect pinned
  // memory pool behavior, see the comments of 'pinned_memory_pool_byte_size_'
  // in 'ServerOptions' for more detail.
  Setting setting_;
  // The setting value.
  std::string value_;
};

//==============================================================================
/// Structure to hold global trace setting for 'ServerOptions' and
/// model-specific trace setting for 'InferOptions'. See here for more
/// information:
/// https://github.com/triton-inference-server/server/blob/main/docs/user_guide/trace.md.
struct Trace {
  enum class Level { OFF, TIMESTAMPS, TENSORS };

  Trace(const std::string& file, const Level& level);

  Trace(
      const std::string& file, const Level& level, const uint32_t rate,
      const int32_t count, const uint32_t log_frequency);

  ~Trace() = default;

  // The file where trace output will be saved. If 'log-frequency' is also
  // specified, this argument value will be the prefix of the files to save the
  // trace output.
  std::string file_;
  // Specify a trace level. OFF to disable tracing, TIMESTAMPS to trace
  // timestamps, TENSORS to trace tensors.
  Level level_;
  // The trace sampling rate. The value represents how many requests will one
  // trace be sampled from. For example, if the trace rate is "1000", 1 trace
  // will be sampled for every 1000 requests. Default is 1000.
  uint32_t rate_;
  // The number of traces to be sampled. If the value is -1, the number of
  // traces to be sampled will not be limited. Default is -1.
  int32_t count_;
  // The trace log frequency. If the value is 0, Triton will only log the trace
  // output to 'file_' when shutting down. Otherwise, Triton will log the trace
  // output to 'file_.<idx>' when it collects the specified number of traces.
  // For example, if the log frequency is 100, when Triton collects the 100-th
  // trace, it logs the traces to file 'file_.0', and when it collects the
  // 200-th trace, it logs the 101-th to the 200-th traces to file file_.1'.
  // Default is 0.
  uint32_t log_frequency_;
};

//==============================================================================
/// Server options that are used to initialize Triton Server.
///
struct ServerOptions {
  ServerOptions(const std::vector<std::string>& model_repository_paths);

  ServerOptions(
      const std::vector<std::string>& model_repository_paths,
      const LoggingOptions& logging, const MetricsOptions& metrics,
      const std::vector<BackendConfig>& be_config, const std::string& server_id,
      const std::string& backend_dir, const std::string& repo_agent_dir,
      const bool disable_auto_complete_config,
      const ModelControlMode& model_control_mode,
      const int32_t repository_poll_secs,
      const std::set<std::string>& startup_models,
      const std::vector<RateLimitResource>& rate_limit_resource,
      const int64_t pinned_memory_pool_byte_size,
      const std::vector<CUDAMemoryPoolByteSize>& cuda_memory_pool_byte_size,
      const uint64_t response_cache_byte_size,
      const double& min_cuda_compute_capability, const bool exit_on_error,
      const int32_t exit_timeout_secs,
      const int32_t buffer_manager_thread_count,
      const uint32_t model_load_thread_count,
      const std::vector<ModelLoadGPULimit>& model_load_gpu_limit,
      const std::vector<HostPolicy>& host_policy, std::shared_ptr<Trace> trace);

  // Paths to model repository directory. Note that if a model is not unique
  // across all model repositories at any time, the model will not be available.
  // See here for more information:
  // https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md.
  std::vector<std::string> model_repository_paths_;
  // Logging options. See the 'LoggingOptions' structure for more information.
  LoggingOptions logging_;
  // Metrics options. See the 'MetricsOptions' structure for more information.
  MetricsOptions metrics_;
  // Backend configuration. See the 'BackendConfig' structure for more
  // information.
  std::vector<BackendConfig> be_config_;
  // The ID of the server.
  std::string server_id_;
  // The global directory searched for backend shared libraries. Default is
  // "/opt/tritonserver/backends". See here for more information:
  // https://github.com/triton-inference-server/backend#backends.
  std::string backend_dir_;
  // The global directory searched for repository agent shared libraries.
  // Default is "/opt/tritonserver/repoagents". See here for more information:
  // https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/repository_agents.md.
  std::string repo_agent_dir_;
  // If set, disables the triton and backends from auto completing model
  // configuration files. Model configuration files must be provided and
  // all required configuration settings must be specified. Default is false.
  // See here for more information:
  // https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#auto-generated-model-configuration.
  bool disable_auto_complete_config_;
  // Specify the mode for model management. Options are "NONE", "POLL" and
  // "EXPLICIT". Default is "NONE". See here for more information:
  // https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md.
  ModelControlMode model_control_mode_;
  // Interval in seconds between each poll of the model repository to check for
  // changes. Valid only when 'model_control_mode_' is set to "POLL". Default
  // is 15.
  int32_t repository_poll_secs_;
  // Specify the the models to be loaded on server startup. This will only take
  // effect if 'model_control_mode_' is set to 'EXPLICIT'.
  std::set<std::string> startup_models_;
  // The number of resources available to the server. Rate limiting is disabled
  // by default, and can be enabled once 'rate_limit_resource_' is set. See the
  // 'RateLimitResource' structure for more information.
  std::vector<RateLimitResource> rate_limit_resource_;
  // The total byte size that can be allocated as pinned system memory. If GPU
  // support is enabled, the server will allocate pinned system memory to
  // accelerate data transfer between host and devices until it exceeds the
  // specified byte size.  If 'NUMA_NODE' is configured via 'host_policy_', the
  // pinned system memory of the pool size will be allocated on each numa node.
  // This option will not affect the allocation conducted by the backend
  // frameworks. Default is 256 MB.
  int64_t pinned_memory_pool_byte_size_;
  // The total byte size that can be allocated as CUDA memory for the GPU
  // device. See the 'CUDAMemoryPoolByteSize' structure for more information.
  std::vector<CUDAMemoryPoolByteSize> cuda_memory_pool_byte_size_;
  // The size in bytes to allocate for a request/response cache. When non-zero,
  // Triton allocates the requested size in CPU memory and shares the cache
  // across all inference requests and across all models. For a given model to
  // use request caching, the model must enable request caching in the model
  // configuration. See here for more information:
  // https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#response-cache.
  // By default, no model uses request caching even if the
  // 'response_cache_byte_size_' is set. Default is 0.
  uint64_t response_cache_byte_size_;
  // The minimum supported CUDA compute capability. GPUs that don't support this
  // compute capability will not be used by the server. Default is 0.
  double min_cuda_compute_capability_;
  // If set, exit the inference server when an error occurs during
  // initialization. Default is true.
  bool exit_on_error_;
  // Timeout (in seconds) when exiting to wait for in-flight inferences to
  // finish. After the timeout expires the server exits even if inferences are
  // still in flight. Default is 30 secs.
  int32_t exit_timeout_secs_;
  // The number of threads used to accelerate copies and other operations
  // required to manage input and output tensor contents. Default is 0.
  int32_t buffer_manager_thread_count_;
  // The number of threads used to concurrently load models in model
  // repositories. Default is 2*<num_cpu_cores>.
  uint32_t model_load_thread_count_;
  // The GPU limit of model loading. See the 'ModelLoadGPULimit' structure for
  // more information.
  std::vector<ModelLoadGPULimit> model_load_gpu_limit_;
  // The host policy setting. See the 'HostPolicy' structure for more
  // information.
  std::vector<HostPolicy> host_policy_;
  // The global trace setting. Default is nullptr, meaning that tracing is not
  // enabled. See the 'Trace' structure for more information.
  std::shared_ptr<Trace> trace_;
};

//==============================================================================
/// Structure to hold repository index for 'ModelIndex' function.
///
struct RepositoryIndex {
  RepositoryIndex(
      const std::string& name, const std::string& version,
      const ModelReadyState& state);

  // The name of the model.
  std::string name_;
  // The version of the model.
  std::string version_;
  // The state of the model. The states are
  // * UNKNOWN: The model is in an unknown state. The model is not available for
  // inferencing.
  // * READY: The model is ready and available for inferencing.
  // * UNAVAILABLE: The model is unavailable, indicating that the model failed
  // to load or has been implicitly or explicitly unloaded. The model is not
  // available for inferencing.
  // * LOADING: The model is being loaded by the inference server. The model is
  // not available for inferencing.
  // * UNLOADING: The model is being unloaded by the inference server. The model
  // is not available for inferencing.
  ModelReadyState state_;
};

//==============================================================================
/// Structure to hold information of a tensor. This object is used for adding
/// input/requested output to an inference request, and retrieving the output
/// result from inference result.
///
struct Tensor {
  Tensor(
      char* buffer, const size_t& byte_size, const DataType& data_type,
      const std::vector<int64_t>& shape, const MemoryType& memory_type,
      const int64_t memory_type_id);

  Tensor(
      char* buffer, const size_t& byte_size, const MemoryType& memory_type,
      const int64_t memory_type_id);

  ~Tensor();

  // The pointer to the start of the buffer.
  char* buffer_;
  // The size of buffer in bytes.
  size_t byte_size_;
  // The data type of the tensor.
  DataType data_type_;
  // The shape of the tensor.
  std::vector<int64_t> shape_;
  // The memory type of the tensor. Valid memory types are "CPU", "CPU_PINNED"
  // and "GPU".
  MemoryType memory_type_;
  // The ID of the memory for the tensor. (e.g. '0' is the memory type id of
  // 'GPU-0')
  int64_t memory_type_id_;

  friend class InternalResult;

 private:
  // Store the custom allocator object in case we need to use it to release
  // the buffer.
  std::shared_ptr<Allocator> custom_allocator_;
  // Indicate if the buffer of this tensor is pre-allocated.
  bool is_pre_alloc_;
  // Indicate if thie tensor is an output from inference.
  bool is_output_;
};

//==============================================================================
/// Structure to hold the full path to the model repository to be registered and
/// the mapping from the original model name to the overriden one. This object
/// is used for calling 'TritonServer::RegisterModelRepo' for registering
/// model repository.
///
struct NewModelRepo {
  NewModelRepo(const std::string& path);

  NewModelRepo(
      const std::string& path, const std::string& original_name,
      const std::string& override_name);

  // The full path to the model repository.
  std::string path_;
  // The original name of the model. This field is optional when there is no
  // name mapping needed.
  std::string original_name_;
  // The original name of the model. This field is optional when there is no
  // name mapping needed.
  std::string override_name_;
};

//==============================================================================
/// Structure to hold options for Inference Request.
///
struct InferOptions {
  InferOptions(const std::string& model_name);

  InferOptions(
      const std::string& model_name, const int64_t model_version,
      const std::string& request_id, const uint64_t correlation_id,
      const std::string& correlation_id_str, const bool sequence_start,
      const bool sequence_end, const uint64_t priority,
      const uint64_t request_timeout,
      std::shared_ptr<Allocator> custom_allocator,
      std::shared_ptr<Trace> trace);

  // The name of the model to run inference.
  std::string model_name_;
  // The version of the model to use while running inference. The default
  // value is "-1" which means the server will select the
  // version of the model based on its internal policy.
  int64_t model_version_;
  // An identifier for the request. If specified will be returned
  // in the response. Default value is an empty string which means no
  // request_id will be used.
  std::string request_id_;
  // The correlation ID of the inference request to be an unsigned integer.
  // Should be used exclusively with 'correlation_id_str_'.
  // Default is 0, which indicates that the request has no correlation ID.
  uint64_t correlation_id_;
  // The correlation ID of the inference request to be a string.
  // Should be used exclusively with 'correlation_id_'.
  // Default value is "".
  std::string correlation_id_str_;
  // Indicates whether the request being added marks the start of the
  // sequence. Default value is False. This argument is ignored if
  // 'sequence_id' is 0.
  bool sequence_start_;
  // Indicates whether the request being added marks the end of the
  // sequence. Default value is False. This argument is ignored if
  // 'sequence_id' is 0.
  bool sequence_end_;
  // Indicates the priority of the request. Priority value zero
  // indicates that the default priority level should be used
  // (i.e. same behavior as not specifying the priority parameter).
  // Lower value priorities indicate higher priority levels. Thus
  // the highest priority level is indicated by setting the parameter
  // to 1, the next highest is 2, etc. If not provided, the server
  // will handle the request using default setting for the model.
  uint64_t priority_;
  // The timeout value for the request, in microseconds. If the request
  // cannot be completed within the time by the server can take a
  // model-specific action such as terminating the request. If not
  // provided, the server will handle the request using default setting
  // for the model.
  uint64_t request_timeout_;
  // User-provided custom reponse allocator object. Default is nullptr.
  // If using custom allocator, the lifetime of this 'Allocator' object should
  // be long enough until `InferResult` object goes out of scope as we need
  // this `Allocator` object to call 'ResponseAllocatorReleaseFn_t' for
  // releasing the response.
  std::shared_ptr<Allocator> custom_allocator_;
  // Update trace setting for the specified model. If not set, will use global
  // trace setting in 'ServerOptions' for tracing if tracing is enabled in
  // 'ServerOptions'. Default is nullptr.
  std::shared_ptr<Trace> trace_;
};

}}}  // namespace triton::developer_tools::server