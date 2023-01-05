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

#include "triton/developer_tools/server_wrapper.h"
#include <stdlib.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "triton/developer_tools/server_wrapper.h"
#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

namespace triton { namespace developer_tools { namespace server {

#define THROW_IF_TRITON_ERR(X)                                     \
  do {                                                             \
    TRITONSERVER_Error* err__ = (X);                               \
    if (err__ != nullptr) {                                        \
      TritonException ex(                                          \
          TRITONSERVER_ErrorCodeString(err__) + std::string("-") + \
          TRITONSERVER_ErrorMessage(err__) + "\n");                \
      TRITONSERVER_ErrorDelete(err__);                             \
      throw ex;                                                    \
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
          TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,                          \
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

using AllocInfo = std::pair<std::shared_ptr<Allocator>, TensorAllocMap>;

//==============================================================================
/// Helper functions
///
std::string
DataTypeString(const DataType& data_type)
{
  switch (data_type) {
    case DataType::BOOL:
      return "BOOL";
    case DataType::UINT8:
      return "UINT8";
    case DataType::UINT16:
      return "UINT16";
    case DataType::UINT32:
      return "UINT32";
    case DataType::UINT64:
      return "UINT64";
    case DataType::INT8:
      return "INT8";
    case DataType::INT16:
      return "INT16";
    case DataType::INT32:
      return "INT32";
    case DataType::INT64:
      return "INT64";
    case DataType::FP16:
      return "FP16";
    case DataType::FP32:
      return "FP32";
    case DataType::FP64:
      return "FP64";
    case DataType::BYTES:
      return "BYTES";
    case DataType::BF16:
      return "BF16";
    default:
      break;
  }

  return "<invalid>";
}

std::string
MemoryTypeString(const MemoryType& memory_type)
{
  switch (memory_type) {
    case MemoryType::CPU:
      return "CPU";
    case MemoryType::CPU_PINNED:
      return "CPU_PINNED";
    case MemoryType::GPU:
      return "GPU";
    default:
      break;
  }

  return "<invalid>";
}

std::string
ModelReadyStateString(const ModelReadyState& state)
{
  switch (state) {
    case ModelReadyState::UNKNOWN:
      return "UNKNOWN";
    case ModelReadyState::READY:
      return "READY";
    case ModelReadyState::UNAVAILABLE:
      return "UNAVAILABLE";
    case ModelReadyState::LOADING:
      return "LOADING";
    case ModelReadyState::UNLOADING:
      return "UNLOADING";
    default:
      return "UNKNOWN";
  }
}

TRITONSERVER_ModelControlMode
ToTritonModelControlMode(const ModelControlMode& mode)
{
  switch (mode) {
    case ModelControlMode::NONE:
      return TRITONSERVER_MODEL_CONTROL_NONE;
    case ModelControlMode::POLL:
      return TRITONSERVER_MODEL_CONTROL_POLL;
    case ModelControlMode::EXPLICIT:
      return TRITONSERVER_MODEL_CONTROL_EXPLICIT;

    default:
      throw TritonException("unsupported model control mode.");
  }
}

TRITONSERVER_LogFormat
ToTritonLogFormat(const LoggingOptions::LogFormat& format)
{
  switch (format) {
    case LoggingOptions::LogFormat::DEFAULT:
      return TRITONSERVER_LOG_DEFAULT;
    case LoggingOptions::LogFormat::ISO8601:
      return TRITONSERVER_LOG_ISO8601;

    default:
      throw TritonException("unsupported log format.");
  }
}

TRITONSERVER_DataType
ToTritonDataType(const DataType& dtype) noexcept
{
  switch (dtype) {
    case DataType::BOOL:
      return TRITONSERVER_TYPE_BOOL;
    case DataType::UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case DataType::UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case DataType::UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case DataType::UINT64:
      return TRITONSERVER_TYPE_UINT64;
    case DataType::INT8:
      return TRITONSERVER_TYPE_INT8;
    case DataType::INT16:
      return TRITONSERVER_TYPE_INT16;
    case DataType::INT32:
      return TRITONSERVER_TYPE_INT32;
    case DataType::INT64:
      return TRITONSERVER_TYPE_INT64;
    case DataType::FP16:
      return TRITONSERVER_TYPE_FP16;
    case DataType::FP32:
      return TRITONSERVER_TYPE_FP32;
    case DataType::FP64:
      return TRITONSERVER_TYPE_FP64;
    case DataType::BYTES:
      return TRITONSERVER_TYPE_BYTES;
    case DataType::BF16:
      return TRITONSERVER_TYPE_BF16;

    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

DataType
TritonToDataType(const TRITONSERVER_DataType& dtype) noexcept
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      return DataType::BOOL;
    case TRITONSERVER_TYPE_UINT8:
      return DataType::UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return DataType::UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return DataType::UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return DataType::UINT64;
    case TRITONSERVER_TYPE_INT8:
      return DataType::INT8;
    case TRITONSERVER_TYPE_INT16:
      return DataType::INT16;
    case TRITONSERVER_TYPE_INT32:
      return DataType::INT32;
    case TRITONSERVER_TYPE_INT64:
      return DataType::INT64;
    case TRITONSERVER_TYPE_FP16:
      return DataType::FP16;
    case TRITONSERVER_TYPE_FP32:
      return DataType::FP32;
    case TRITONSERVER_TYPE_FP64:
      return DataType::FP64;
    case TRITONSERVER_TYPE_BYTES:
      return DataType::BYTES;
    case TRITONSERVER_TYPE_BF16:
      return DataType::BF16;

    default:
      return DataType::INVALID;
  }
}

TRITONSERVER_MemoryType
ToTritonMemoryType(const MemoryType& mem_type)
{
  switch (mem_type) {
    case MemoryType::CPU:
      return TRITONSERVER_MEMORY_CPU;
    case MemoryType::CPU_PINNED:
      return TRITONSERVER_MEMORY_CPU_PINNED;
    case MemoryType::GPU:
      return TRITONSERVER_MEMORY_GPU;

    default:
      throw TritonException("unsupported memory type.");
  }
}

MemoryType
TritonToMemoryType(const TRITONSERVER_MemoryType& mem_type)
{
  switch (mem_type) {
    case TRITONSERVER_MEMORY_CPU:
      return MemoryType::CPU;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return MemoryType::CPU_PINNED;
    case TRITONSERVER_MEMORY_GPU:
      return MemoryType::GPU;

    default:
      throw TritonException("unsupported memory type.");
  }
}

ModelReadyState
StringToModelReadyState(const std::string& state) noexcept
{
  if (state == "UNKNOWN") {
    return ModelReadyState::UNKNOWN;
  } else if (state == "READY") {
    return ModelReadyState::READY;
  } else if (state == "UNAVAILABLE") {
    return ModelReadyState::UNAVAILABLE;
  } else if (state == "LOADING") {
    return ModelReadyState::LOADING;
  } else if (state == "UNLOADING") {
    return ModelReadyState::UNLOADING;
  } else {
    return ModelReadyState::UNKNOWN;
  }
}

std::string
HostPolicySettingString(const HostPolicy::Setting& setting)
{
  switch (setting) {
    case HostPolicy::Setting::NUMA_NODE:
      return "numa-node";
    case HostPolicy::Setting::CPU_CORES:
      return "cpu-cores";

    default:
      throw TritonException("unsupported host policy setting.");
  }
}

TRITONSERVER_InferenceTraceLevel
ToTritonTraceLevel(const Trace::Level& level)
{
  switch (level) {
    case Trace::Level::OFF:
      return TRITONSERVER_TRACE_LEVEL_DISABLED;
    case Trace::Level::TIMESTAMPS:
      return TRITONSERVER_TRACE_LEVEL_TIMESTAMPS;
    case Trace::Level::TENSORS:
      return TRITONSERVER_TRACE_LEVEL_TENSORS;

    default:
      throw TritonException("unsupported trace level.");
  }
}

//==============================================================================
/// Structure to hold response parameters for InfeResult object. The kinds
/// of parameters in a response can be created by the backend side using
/// 'TRITONBACKEND_ResponseSet*Parameter' APIs.
/// See here for more information:
/// https://github.com/triton-inference-server/backend/tree/main/examples#add-key-value-parameters-to-a-response
struct ResponseParameters {
  explicit ResponseParameters(
      const char* name, TRITONSERVER_ParameterType type, const void* vvalue)
      : name_(name), type_(type), vvalue_(vvalue)
  {
  }

  // The name of the parameter.
  const char* name_;
  // The type of the parameter. Valid types are TRITONSERVER_PARAMETER_STRING,
  // TRITONSERVER_PARAMETER_INT, TRITONSERVER_PARAMETER_BOOL, and
  // TRITONSERVER_PARAMETER_BYTES.
  TRITONSERVER_ParameterType type_;
  // The pointer to the parameter value.
  const void* vvalue_;
};

//==============================================================================
/// InternalServer class
///
class InternalServer : public TritonServer {
 public:
  InternalServer(const ServerOptions& server_options);

  ~InternalServer();

  static TRITONSERVER_Error* ResponseAlloc(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id);
  static TRITONSERVER_Error* ResponseRelease(
      TRITONSERVER_ResponseAllocator* allocator, void* buffer,
      void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

  static void InferResponseComplete(
      TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      void* userp);
  static void InferRequestComplete(
      TRITONSERVER_InferenceRequest* request, const uint32_t flags,
      void* userp);

  // START_JAVA_CUSTOM_FUNCTIONS
  std::future<std::unique_ptr<InferResult>> AsyncInfer(
      InferRequest& infer_request) override;
  // END_JAVA_CUSTOM_FUNCTIONS

 private:
  void StartRepoPollThread();
  void StopRepoPollThread();

  bool is_exiting_;
  std::mutex exit_mu_;
  std::condition_variable exit_cv_;
  int32_t repository_poll_secs_;
  std::thread repo_poll_thread_;
};

//==============================================================================
/// InternalRequest class
///
class InternalRequest : public InferRequest {
 public:
  InternalRequest(const InferOptions& infer_options);

  ~InternalRequest();

  static std::shared_ptr<Allocator> custom_allocator_;
  static TRITONSERVER_ResponseAllocator* custom_triton_allocator_;
};

std::shared_ptr<Allocator> InternalRequest::custom_allocator_;
TRITONSERVER_ResponseAllocator* InternalRequest::custom_triton_allocator_;

//==============================================================================
/// InternalResult class
///
class InternalResult : public InferResult {
 public:
  InternalResult();

  ~InternalResult();

  void FinalizeResponse(
      TRITONSERVER_InferenceResponse* response, const AllocInfo& alloc_info);
};

//==============================================================================
/// Default functions for allocator.
///
TRITONSERVER_Error*
InternalServer::ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  auto p = reinterpret_cast<InferRequest*>(userp);
  if ((p->tensor_alloc_map_.find(tensor_name) != p->tensor_alloc_map_.end() &&
       std::get<0>(p->tensor_alloc_map_[tensor_name]) != nullptr)) {
    if (byte_size != std::get<1>(p->tensor_alloc_map_[tensor_name])) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "Unexpected byte-size of pre-allocated buffer for '" +
              std::string(tensor_name) + "'. Expected " +
              std::to_string(byte_size) + ", got " +
              std::to_string(std::get<1>(p->tensor_alloc_map_[tensor_name])))
              .c_str());
    }

    // Use the pre-allocated buffer.
    *buffer = const_cast<void*>(std::get<0>(p->tensor_alloc_map_[tensor_name]));

    *actual_memory_type = std::get<2>(p->tensor_alloc_map_[tensor_name]);
    *actual_memory_type_id = std::get<3>(p->tensor_alloc_map_[tensor_name]);
  } else {
    // Initially attempt to make the actual memory type and id that we
    // allocate be the same as preferred memory type
    *actual_memory_type = preferred_memory_type;
    *actual_memory_type_id = preferred_memory_type_id;

    // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
    // need to do any other book-keeping.
    if (byte_size == 0) {
      *buffer = nullptr;
      *buffer_userp = nullptr;
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE, ("allocated " + std::to_string(byte_size) +
                                     " bytes for result tensor " + tensor_name)
                                        .c_str());
    } else {
      void* allocated_ptr = nullptr;
      switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
        case TRITONSERVER_MEMORY_CPU_PINNED: {
          auto err = cudaSetDevice(*actual_memory_type_id);
          if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
              (err != cudaErrorInsufficientDriver)) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "unable to recover current CUDA device: " +
                    std::string(cudaGetErrorString(err)))
                    .c_str());
          }

          err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
          if (err != cudaSuccess) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "cudaHostAlloc failed: " +
                    std::string(cudaGetErrorString(err)))
                    .c_str());
          }
          break;
        }

        case TRITONSERVER_MEMORY_GPU: {
          auto err = cudaSetDevice(*actual_memory_type_id);
          if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
              (err != cudaErrorInsufficientDriver)) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "unable to recover current CUDA device: " +
                    std::string(cudaGetErrorString(err)))
                    .c_str());
          }

          err = cudaMalloc(&allocated_ptr, byte_size);
          if (err != cudaSuccess) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "cudaMalloc failed: " +
                    std::string(cudaGetErrorString(err)))
                    .c_str());
          }
          break;
        }
#endif  // TRITON_ENABLE_GPU

        // Use CPU memory if the requested memory type is unknown
        // (default case).
        case TRITONSERVER_MEMORY_CPU:
        default: {
          *actual_memory_type = TRITONSERVER_MEMORY_CPU;
          allocated_ptr = malloc(byte_size);
          break;
        }
      }

      // Pass the tensor name with buffer_userp so we can show it when
      // releasing the buffer.
      if (allocated_ptr != nullptr) {
        *buffer = allocated_ptr;
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            ("allocated " + std::to_string(byte_size) + " bytes in " +
             TRITONSERVER_MemoryTypeString(*actual_memory_type) +
             " for result tensor " + tensor_name)
                .c_str());
      }
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
InternalServer::ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  // Do nothing here as the destructor of the output tensor will release the
  // output buffer.
  return nullptr;  // Success
}

TRITONSERVER_Error*
CustomStartFn(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  if (InternalRequest::custom_allocator_->StartFn() != nullptr) {
    try {
      InternalRequest::custom_allocator_->StartFn()(userp);
    }
    catch (const TritonException& ex) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, ex.what());
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
CustomAllocFn(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  if (InternalRequest::custom_allocator_->AllocFn() != nullptr) {
    try {
      MemoryType preferred_mem_type = TritonToMemoryType(preferred_memory_type);
      MemoryType actual_mem_type;
      InternalRequest::custom_allocator_->AllocFn()(
          tensor_name, byte_size, preferred_mem_type, preferred_memory_type_id,
          buffer, &actual_mem_type, actual_memory_type_id);

      *actual_memory_type = ToTritonMemoryType(actual_mem_type);
    }
    catch (const TritonException& ex) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, ex.what());
    }
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Custom response allocation function is not set.");
  }
  *buffer_userp = nullptr;

  return nullptr;  // Success
}

void
InternalServer::InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");
  }
}

void
InternalServer::InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  auto p = reinterpret_cast<InferRequest*>(userp);
  // The allocation info of output tensor, which will be used to finalize
  // the reponse and stored in the ouput 'Tensor' object so that When calling
  // the destructor of an output tensor, it will know how to clean the buffer
  // correctly.
  AllocInfo alloc_info(
      p->infer_options_->custom_allocator_, p->tensor_alloc_map_);
  bool is_decoupled = p->is_decoupled_;

  if (response != nullptr) {
    std::unique_ptr<InternalResult> result = std::make_unique<InternalResult>();
    result->FinalizeResponse(response, alloc_info);
    std::unique_ptr<InferResult> infer_result = std::move(result);

    if (!is_decoupled) {
      infer_result->next_result_future_.reset();
      p->prev_promise_->set_value(std::move(infer_result));
      p->prev_promise_.reset();
    } else {
      if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
        // Not the last reponse. Need to store the promise associated with the
        // next future.
        auto promise = new std::promise<std::unique_ptr<InferResult>>();
        infer_result->next_result_future_ =
            std::make_unique<std::future<std::unique_ptr<InferResult>>>(
                promise->get_future());
        p->prev_promise_->set_value(std::move(infer_result));
        p->prev_promise_.reset(std::move(promise));
      } else {
        // The last response.
        infer_result->next_result_future_.reset();
        p->prev_promise_->set_value(std::move(infer_result));
        p->prev_promise_.reset();
      }
    }
  } else if (
      is_decoupled && (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0) {
    // An empty response may be the last reponse for decoupled models.
    p->prev_promise_->set_value(nullptr);
    p->prev_promise_.reset();
  } else {
    p->prev_promise_->set_value(nullptr);
    p->prev_promise_.reset();
    throw TritonException("Unexpected empty response.");
  }
}

TRITONSERVER_Error*
OutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  // Always attempt to return the memory in the requested memory_type and
  // memory_type_id when using default allocator.
  return nullptr;  // Success
}

LoggingOptions::LoggingOptions()
    : verbose_(VerboseLevel::OFF), info_(true), warn_(true), error_(true),
      format_(LogFormat::DEFAULT), log_file_("")
{
}

LoggingOptions::LoggingOptions(
    const VerboseLevel verbose, const bool info, const bool warn,
    const bool error, const LogFormat& format, const std::string& log_file)
    : info_(info), warn_(warn), error_(error), format_(format),
      log_file_(log_file)
{
  if ((verbose < VerboseLevel::MIN) || (verbose > VerboseLevel::MAX)) {
    verbose_ = VerboseLevel::OFF;
  } else {
    verbose_ = verbose;
  }
}

MetricsOptions::MetricsOptions()
    : allow_metrics_(true), allow_gpu_metrics_(true), allow_cpu_metrics_(true),
      metrics_interval_ms_(2000)
{
}

MetricsOptions::MetricsOptions(
    const bool allow_metrics, const bool allow_gpu_metrics,
    const bool allow_cpu_metrics, const uint64_t metrics_interval_ms)
    : allow_metrics_(allow_metrics), allow_gpu_metrics_(allow_gpu_metrics),
      allow_cpu_metrics_(allow_cpu_metrics),
      metrics_interval_ms_(metrics_interval_ms)
{
}

BackendConfig::BackendConfig(
    const std::string& name, const std::string& setting,
    const std::string& value)
    : name_(name), setting_(setting), value_(value)
{
}

RateLimitResource::RateLimitResource(const std::string& name, const int count)
    : name_(name), count_(count), device_(-1)
{
}

RateLimitResource::RateLimitResource(
    const std::string& name, const int count, const int device)
    : name_(name), count_(count), device_(device)
{
}

CUDAMemoryPoolByteSize::CUDAMemoryPoolByteSize(
    const int gpu_device, const uint64_t size)
    : gpu_device_(gpu_device), size_(size)
{
}

ModelLoadGPULimit::ModelLoadGPULimit(
    const int device_id, const double& fraction)
    : device_id_(device_id), fraction_(fraction)
{
}

HostPolicy::HostPolicy(
    const std::string& name, const Setting& setting, const std::string& value)
    : name_(name), setting_(setting), value_(value)
{
}

Trace::Trace(const std::string& file, const Level& level)
    : file_(file), level_(level), rate_(1000), count_(-1), log_frequency_(0)
{
}

Trace::Trace(
    const std::string& file, const Level& level, const uint32_t rate,
    const int32_t count, const uint32_t log_frequency)
    : file_(file), level_(level), rate_(rate), count_(count),
      log_frequency_(log_frequency)
{
}

ServerOptions::ServerOptions(
    const std::vector<std::string>& model_repository_paths)
    : model_repository_paths_(model_repository_paths),
      logging_(LoggingOptions()), metrics_(MetricsOptions()),
      server_id_("triton"), backend_dir_("/opt/tritonserver/backends"),
      repo_agent_dir_("/opt/tritonserver/repoagents"),
      disable_auto_complete_config_(false),
      model_control_mode_(ModelControlMode::NONE), repository_poll_secs_(15),
      pinned_memory_pool_byte_size_(1 << 28), response_cache_byte_size_(0),
      min_cuda_compute_capability_(0), exit_on_error_(true),
      exit_timeout_secs_(30), buffer_manager_thread_count_(0),
      model_load_thread_count_(
          std::max(2u, 2 * std::thread::hardware_concurrency())),
      trace_(nullptr)
{
  // FIXME: Use iterator instead of vector for 'model_repository_paths_'.
  be_config_.clear();
  startup_models_.clear();
  rate_limit_resource_.clear();
  cuda_memory_pool_byte_size_.clear();
  model_load_gpu_limit_.clear();
  host_policy_.clear();
}

ServerOptions::ServerOptions(
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
    const int32_t exit_timeout_secs, const int32_t buffer_manager_thread_count,
    const uint32_t model_load_thread_count,
    const std::vector<ModelLoadGPULimit>& model_load_gpu_limit,
    const std::vector<HostPolicy>& host_policy, std::shared_ptr<Trace> trace)
    : model_repository_paths_(model_repository_paths), logging_(logging),
      metrics_(metrics), be_config_(be_config), server_id_(server_id),
      backend_dir_(backend_dir), repo_agent_dir_(repo_agent_dir),
      disable_auto_complete_config_(disable_auto_complete_config),
      model_control_mode_(model_control_mode),
      repository_poll_secs_(repository_poll_secs),
      startup_models_(startup_models),
      rate_limit_resource_(rate_limit_resource),
      pinned_memory_pool_byte_size_(pinned_memory_pool_byte_size),
      cuda_memory_pool_byte_size_(cuda_memory_pool_byte_size),
      response_cache_byte_size_(response_cache_byte_size),
      min_cuda_compute_capability_(min_cuda_compute_capability),
      exit_on_error_(exit_on_error), exit_timeout_secs_(exit_timeout_secs),
      buffer_manager_thread_count_(buffer_manager_thread_count),
      model_load_thread_count_(model_load_thread_count),
      model_load_gpu_limit_(model_load_gpu_limit), host_policy_(host_policy),
      trace_(trace)
{
}

RepositoryIndex::RepositoryIndex(
    const std::string& name, const std::string& version,
    const ModelReadyState& state)
    : name_(name), version_(version), state_(state)
{
}

Tensor::Tensor(
    char* buffer, const size_t& byte_size, const DataType& data_type,
    const std::vector<int64_t>& shape, const MemoryType& memory_type,
    const int64_t memory_type_id)
    : buffer_(buffer), byte_size_(byte_size), data_type_(data_type),
      shape_(shape), memory_type_(memory_type), memory_type_id_(memory_type_id),
      is_pre_alloc_(false), is_output_(false)
{
  custom_allocator_.reset();
}

Tensor::Tensor(
    char* buffer, const size_t& byte_size, const MemoryType& memory_type,
    const int64_t memory_type_id)
    : buffer_(buffer), byte_size_(byte_size), data_type_(DataType::INVALID),
      shape_({}), memory_type_(memory_type), memory_type_id_(memory_type_id),
      is_pre_alloc_(false), is_output_(false)
{
  custom_allocator_.reset();
}

Tensor::~Tensor()
{
  // No need to clean the buffer for output tesnsor with pre-allocated buffer
  // and input tensor.
  if (!is_pre_alloc_ && is_output_) {
    if (custom_allocator_ == nullptr) {
      std::stringstream ss;
      ss << (void*)buffer_;
      std::string buffer_str = ss.str();

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          ("Releasing buffer " + buffer_str + " of size " +
           std::to_string(byte_size_) + " in " + MemoryTypeString(memory_type_))
              .c_str());

      switch (memory_type_) {
        case MemoryType::CPU:
          free(buffer_);
          break;
#ifdef TRITON_ENABLE_GPU
        case MemoryType::CPU_PINNED: {
          auto err = cudaSetDevice(memory_type_id_);
          if (err == cudaSuccess) {
            err = cudaFreeHost(buffer_);
          }
          if (err != cudaSuccess) {
            std::cerr << "error: failed to cudaFree " << buffer_ << ": "
                      << cudaGetErrorString(err) << std::endl;
          }
          break;
        }
        case MemoryType::GPU: {
          auto err = cudaSetDevice(memory_type_id_);
          if (err == cudaSuccess) {
            err = cudaFree(buffer_);
          }
          if (err != cudaSuccess) {
            std::cerr << "error: failed to cudaFree " << buffer_ << ": "
                      << cudaGetErrorString(err) << std::endl;
          }
          break;
        }
#endif  // TRITON_ENABLE_GPU
        default:
          std::cerr
              << "error: unexpected buffer allocated in CUDA managed memory"
              << std::endl;
          break;
      }
    } else {
      if (custom_allocator_->ReleaseFn() == nullptr) {
        std::cerr << "error: ReleaseFn() is not set in custom allocator."
                  << std::endl;
      } else {
        try {
          custom_allocator_->ReleaseFn()(
              reinterpret_cast<void*>(buffer_), byte_size_, memory_type_,
              memory_type_id_);
        }
        catch (const TritonException& ex) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("error: using custom allocator - ") + ex.what())
                  .c_str());
        }
      }
    }
  }
}

NewModelRepo::NewModelRepo(const std::string& path)
    : path_(path), original_name_(""), override_name_("")
{
}

NewModelRepo::NewModelRepo(
    const std::string& path, const std::string& original_name,
    const std::string& override_name)
    : path_(path), original_name_(original_name), override_name_(override_name)
{
}

InferOptions::InferOptions(const std::string& model_name)
    : model_name_(model_name), model_version_(-1), request_id_(""),
      correlation_id_(0), correlation_id_str_(""), sequence_start_(false),
      sequence_end_(false), priority_(0), request_timeout_(0),
      custom_allocator_(nullptr), trace_(nullptr)
{
}

InferOptions::InferOptions(
    const std::string& model_name, const int64_t model_version,
    const std::string& request_id, const uint64_t correlation_id,
    const std::string& correlation_id_str, const bool sequence_start,
    const bool sequence_end, const uint64_t priority,
    const uint64_t request_timeout, std::shared_ptr<Allocator> custom_allocator,
    std::shared_ptr<Trace> trace)
    : model_name_(model_name), model_version_(model_version),
      request_id_(request_id), correlation_id_(correlation_id),
      correlation_id_str_(correlation_id_str), sequence_start_(sequence_start),
      sequence_end_(sequence_end), priority_(priority),
      request_timeout_(request_timeout), custom_allocator_(custom_allocator),
      trace_(trace)
{
}

std::unique_ptr<GenericTritonServer>
GenericTritonServer::Create(const ServerOptions& server_options)
{
  return TritonServer::Create(server_options);
}

GenericTritonServer::~GenericTritonServer() {}


std::unique_ptr<TritonServer>
TritonServer::Create(const ServerOptions& options)
{
  std::unique_ptr<InternalServer> internal_server;
  internal_server.reset(new InternalServer(options));
  return internal_server;
}

TritonServer::~TritonServer() {}

void
TritonServer::LoadModel(const std::string& model_name)
{
  try {
    THROW_IF_TRITON_ERR(
        TRITONSERVER_ServerLoadModel(server_.get(), model_name.c_str()));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - LoadModel: ") + ex.what());
  }
}

void
TritonServer::UnloadModel(const std::string& model_name)
{
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerUnloadModelAndDependents(
        server_.get(), model_name.c_str()));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - UnloadModel: ") + ex.what());
  }
}

std::set<std::string>
TritonServer::LoadedModels()
{
  try {
    std::vector<RepositoryIndex> repository_index = ModelIndex();
    std::set<std::string> models;
    for (size_t i = 0; i < repository_index.size(); i++) {
      models.insert(repository_index[i].name_);
    }
    return models;
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - LoadedModels: ") + ex.what());
  }
}

std::vector<RepositoryIndex>
TritonServer::ModelIndex()
{
  std::vector<RepositoryIndex> repository_index;
  TRITONSERVER_Message* message = nullptr;
  uint32_t flags = TRITONSERVER_INDEX_FLAG_READY;
  try {
    THROW_IF_TRITON_ERR(
        TRITONSERVER_ServerModelIndex(server_.get(), flags, &message));
    const char* buffer;
    size_t byte_size;
    THROW_IF_TRITON_ERR(
        TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size));

    common::TritonJson::Value repo_index;
    THROW_IF_TRITON_ERR(repo_index.Parse(buffer, byte_size));
    THROW_IF_TRITON_ERR(TRITONSERVER_MessageDelete(message));

    for (size_t i = 0; i < repo_index.ArraySize(); i++) {
      triton::common::TritonJson::Value index;
      THROW_IF_TRITON_ERR(repo_index.IndexAsObject(i, &index));
      std::string name, version, state;
      THROW_IF_TRITON_ERR(index.MemberAsString("name", &name));
      THROW_IF_TRITON_ERR(index.MemberAsString("version", &version));
      THROW_IF_TRITON_ERR(index.MemberAsString("state", &state));
      repository_index.push_back(
          RepositoryIndex(name, version, StringToModelReadyState(state)));
    }
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - ModelIndex: ") + ex.what());
  }

  return repository_index;
}

std::string
TritonServer::ServerMetrics()
{
  std::string metrics_str = "";
  TRITONSERVER_Metrics* metrics = nullptr;
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerMetrics(server_.get(), &metrics));
    const char* base;
    size_t byte_size;
    THROW_IF_TRITON_ERR(TRITONSERVER_MetricsFormatted(
        metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size));
    metrics_str = std::string(base, byte_size);
    THROW_IF_TRITON_ERR(TRITONSERVER_MetricsDelete(metrics));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - Metrics: ") + ex.what());
  }

  return metrics_str;
}

std::string
TritonServer::ModelStatistics(
    const std::string& model_name, const int64_t model_version)
{
  TRITONSERVER_Message* model_stats = nullptr;
  std::string metrics_str = "";
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerModelStatistics(
        server_.get(), model_name.c_str(), model_version, &model_stats));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - ModelStatistics: ") + ex.what());
  }
  const char* base;
  size_t byte_size;
  try {
    THROW_IF_TRITON_ERR(
        TRITONSERVER_MessageSerializeToJson(model_stats, &base, &byte_size));
    metrics_str = std::string(base, byte_size);
    THROW_IF_TRITON_ERR(TRITONSERVER_MessageDelete(model_stats));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - ModelStatistics: ") + ex.what());
  }

  return metrics_str;
}

bool
TritonServer::IsServerLive()
{
  bool live = false;
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerIsLive(server_.get(), &live));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - IsLive: ") + ex.what());
  }

  return live;
}

bool
TritonServer::IsServerReady()
{
  bool ready = false;
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerIsReady(server_.get(), &ready));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - IsReady: ") + ex.what());
  }

  return ready;
}

void
TritonServer::ServerStop()
{
  TRITONSERVER_ServerStop(server_.get());
}

bool
TritonServer::IsModelReady(
    const std::string& model_name, const int64_t model_version)
{
  bool ready = false;
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerModelIsReady(
        server_.get(), model_name.c_str(), model_version, &ready));
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - IsModelReady: ") + ex.what());
  }

  return ready;
}

std::string
TritonServer::ModelConfig(
    const std::string& model_name, const int64_t model_version)
{
  std::string config_str = "";
  TRITONSERVER_Message* model_config = nullptr;
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerModelConfig(
        server_.get(), model_name.c_str(), model_version,
        1 /* config_version */, &model_config));
    const char* base;
    size_t byte_size;
    THROW_IF_TRITON_ERR(
        TRITONSERVER_MessageSerializeToJson(model_config, &base, &byte_size));
    config_str = std::string(base, byte_size);
    TRITONSERVER_MessageDelete(model_config);
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - ModelConfig: ") + ex.what());
  }

  return config_str;
}

std::string
TritonServer::ServerMetadata()
{
  std::string metadata_str = "";
  TRITONSERVER_Message* server_metadata = nullptr;
  try {
    THROW_IF_TRITON_ERR(
        TRITONSERVER_ServerMetadata(server_.get(), &server_metadata));
    const char* base;
    size_t byte_size;
    THROW_IF_TRITON_ERR(TRITONSERVER_MessageSerializeToJson(
        server_metadata, &base, &byte_size));
    metadata_str = std::string(base, byte_size);
    TRITONSERVER_MessageDelete(server_metadata);
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - ModelConfig: ") + ex.what());
  }

  return metadata_str;
}

std::string
TritonServer::ModelMetadata(
    const std::string& model_name, const int64_t model_version)
{
  std::string metadata_str = "";
  TRITONSERVER_Message* model_metadata = nullptr;
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerModelMetadata(
        server_.get(), model_name.c_str(), model_version, &model_metadata));
    const char* base;
    size_t byte_size;
    THROW_IF_TRITON_ERR(
        TRITONSERVER_MessageSerializeToJson(model_metadata, &base, &byte_size));
    metadata_str = std::string(base, byte_size);
    TRITONSERVER_MessageDelete(model_metadata);
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - ModelMetadata: ") + ex.what());
  }

  return metadata_str;
}

void
TritonServer::RegisterModelRepo(const NewModelRepo& new_model_repo)
{
  try {
    if (new_model_repo.original_name_.empty() ||
        new_model_repo.override_name_.empty()) {
      // Register without name mapping.
      THROW_IF_TRITON_ERR(TRITONSERVER_ServerRegisterModelRepository(
          server_.get(), new_model_repo.path_.c_str(), nullptr, 0));
    } else {
      std::shared_ptr<TRITONSERVER_Parameter> managed_param(
          TRITONSERVER_ParameterNew(
              new_model_repo.original_name_.c_str(),
              TRITONSERVER_PARAMETER_STRING,
              new_model_repo.override_name_.c_str()),
          TRITONSERVER_ParameterDelete);
      std::vector<const TRITONSERVER_Parameter*> name_map{managed_param.get()};
      THROW_IF_TRITON_ERR(TRITONSERVER_ServerRegisterModelRepository(
          server_.get(), new_model_repo.path_.c_str(), name_map.data(),
          name_map.size()));
    }
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - RegisterModelRepo: ") + ex.what());
  }
}

void
TritonServer::UnregisterModelRepo(const std::string& repo_path)
{
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerUnregisterModelRepository(
        server_.get(), repo_path.c_str()));
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - UnregisterModelRepo: ") + ex.what());
  }
}

void
TritonServer::PrepareInferenceRequest(
    TRITONSERVER_InferenceRequest** irequest, const InferRequest& request)
{
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestNew(
        irequest, server_.get(), request.infer_options_->model_name_.c_str(),
        request.infer_options_->model_version_));

    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetId(
        *irequest, request.infer_options_->request_id_.c_str()));
    if (request.infer_options_->correlation_id_str_.empty()) {
      THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetCorrelationId(
          *irequest, request.infer_options_->correlation_id_));
    } else {
      THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetCorrelationIdString(
          *irequest, request.infer_options_->correlation_id_str_.c_str()));
    }

    uint32_t flags = 0;
    if (request.infer_options_->sequence_start_) {
      flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
    }
    if (request.infer_options_->sequence_end_) {
      flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
    }
    THROW_IF_TRITON_ERR(
        TRITONSERVER_InferenceRequestSetFlags(*irequest, flags));

    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetPriority(
        *irequest, request.infer_options_->priority_));

    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
        *irequest, request.infer_options_->request_timeout_));
    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        *irequest, InternalServer::InferRequestComplete,
        nullptr /* request_release_userp */));
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - PrepareInferenceRequest: ") + ex.what());
  }
}

void
TritonServer::PrepareInferenceInput(
    TRITONSERVER_InferenceRequest* irequest, const InferRequest& request)
{
  try {
    for (auto& input : request.inputs_) {
      THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestAddInput(
          irequest, input.first.c_str(),
          ToTritonDataType(input.second->data_type_),
          input.second->shape_.data(), input.second->shape_.size()));

      TRITONSERVER_MemoryType memory_type =
          ToTritonMemoryType(input.second->memory_type_);
      THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestAppendInputData(
          irequest, input.first.c_str(), input.second->buffer_,
          input.second->byte_size_, memory_type,
          input.second->memory_type_id_));
    }
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - PrepareInferenceInput: ") + ex.what());
  }
}

void
TritonServer::PrepareInferenceOutput(
    TRITONSERVER_InferenceRequest* irequest, InferRequest& request)
{
  try {
    for (auto& infer_output : request.outputs_) {
      const char* name = infer_output->Name().c_str();
      THROW_IF_TRITON_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, name));
      if (infer_output->Buffer() != nullptr) {
        request.tensor_alloc_map_[name] = std::make_tuple(
            infer_output->Buffer(), infer_output->ByteSize(),
            ToTritonMemoryType(infer_output->GetMemoryType()),
            infer_output->MemoryTypeId());
      }
    }
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - PrepareInferenceOutput: ") + ex.what());
  }
}

void
TritonServer::AsyncInferHelper(
    TRITONSERVER_InferenceRequest** irequest, const InferRequest& infer_request)
{
  bool is_ready = false;
  std::string model_name = infer_request.infer_options_->model_name_.c_str();
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerModelIsReady(
      server_.get(), model_name.c_str(),
      infer_request.infer_options_->model_version_, &is_ready));

  if (!is_ready) {
    throw TritonException(
        (std::string("Failed for execute the inference request. Model '") +
         model_name + "' is not ready.")
            .c_str());
  }

  PrepareInferenceRequest(irequest, infer_request);
  PrepareInferenceInput(*irequest, infer_request);
  PrepareInferenceOutput(*irequest, const_cast<InferRequest&>(infer_request));
}

InternalServer::InternalServer(const ServerOptions& options)
    : is_exiting_(false)
{
  uint32_t api_version_major, api_version_minor;
  THROW_IF_TRITON_ERR(
      TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor));
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    throw TritonException("triton server API version mismatch");
  }

  TRITONSERVER_ServerOptions* server_options = nullptr;
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsNew(&server_options));

  // Set model_repository_path
  for (const auto& model_repository_path : options.model_repository_paths_) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(
        server_options, model_repository_path.c_str()));
  }

  // Set logging options
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(
      server_options, static_cast<int>(options.logging_.verbose_)));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogInfo(
      server_options, options.logging_.info_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogWarn(
      server_options, options.logging_.warn_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogError(
      server_options, options.logging_.error_));
  TRITONSERVER_LogFormat log_format =
      ToTritonLogFormat(options.logging_.format_);
  THROW_IF_TRITON_ERR(
      TRITONSERVER_ServerOptionsSetLogFormat(server_options, log_format));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogFile(
      server_options, options.logging_.log_file_.c_str()));

  // Set metrics options
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetMetrics(
      server_options, options.metrics_.allow_metrics_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetGpuMetrics(
      server_options, options.metrics_.allow_gpu_metrics_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetCpuMetrics(
      server_options, options.metrics_.allow_cpu_metrics_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetMetricsInterval(
      server_options, options.metrics_.metrics_interval_ms_));

  // Set backend configuration
  for (const auto& bc : options.be_config_) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetBackendConfig(
        server_options, bc.name_.c_str(), bc.setting_.c_str(),
        bc.value_.c_str()));
  }

  // Set server id
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetServerId(
      server_options, options.server_id_.c_str()));

  // Set backend directory
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(
      server_options, options.backend_dir_.c_str()));

  // Set repo agent directory
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
      server_options, options.repo_agent_dir_.c_str()));

  // Set auto-complete model config
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetStrictModelConfig(
      server_options, options.disable_auto_complete_config_));

  // Set model control mode
  TRITONSERVER_ModelControlMode model_control_mode =
      ToTritonModelControlMode(options.model_control_mode_);
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetModelControlMode(
      server_options, model_control_mode));

  if (model_control_mode == TRITONSERVER_MODEL_CONTROL_POLL) {
    repository_poll_secs_ = std::max(0, options.repository_poll_secs_);
  } else {
    repository_poll_secs_ = 0;
  }

  // Set startup models
  for (const auto& model : options.startup_models_) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetStartupModel(
        server_options, model.c_str()));
  }

  // Set rate limit mode and resource
  if (!options.rate_limit_resource_.empty()) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetRateLimiterMode(
        server_options, TRITONSERVER_RATE_LIMIT_EXEC_COUNT));
    for (const auto& resource : options.rate_limit_resource_) {
      THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsAddRateLimiterResource(
          server_options, resource.name_.c_str(), resource.count_,
          resource.device_));
    }
  } else {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetRateLimiterMode(
        server_options, TRITONSERVER_RATE_LIMIT_OFF));
  }

  // Set pinned memory pool byte size
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
      server_options, options.pinned_memory_pool_byte_size_));

  // Set CUDA memory pool byte size
  for (const auto& cuda_pool : options.cuda_memory_pool_byte_size_) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
        server_options, cuda_pool.gpu_device_, cuda_pool.size_));
  }

  // Set response cache byte size
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetResponseCacheByteSize(
      server_options, options.response_cache_byte_size_));

  // Set minimum supported CUDA compute capability
  THROW_IF_TRITON_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          server_options, options.min_cuda_compute_capability_));

  // Set exit on error
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetExitOnError(
      server_options, options.exit_on_error_));

  // Set exit timeout
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetExitTimeout(
      server_options, std::max(0, options.exit_timeout_secs_)));

  // Set buffer manager thread count
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
      server_options, std::max(0, options.buffer_manager_thread_count_)));

  // Set model load thread count
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
      server_options, std::max(1u, options.model_load_thread_count_)));

  // Set model load device limit
  for (const auto& limit : options.model_load_gpu_limit_) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
        server_options, TRITONSERVER_INSTANCEGROUPKIND_GPU, limit.device_id_,
        limit.fraction_));
  }

  // Set host policy
  for (const auto& hp : options.host_policy_) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetHostPolicy(
        server_options, hp.name_.c_str(),
        HostPolicySettingString(hp.setting_).c_str(), hp.value_.c_str()));
  }

  TRITONSERVER_Server* server_ptr;
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsDelete(server_options));
  server_ = std::shared_ptr<TRITONSERVER_Server>(
      server_ptr, TRITONSERVER_ServerDelete);

  // Initialize allocator
  allocator_ = nullptr;
  THROW_IF_TRITON_ERR(TRITONSERVER_ResponseAllocatorNew(
      &allocator_, InternalServer::ResponseAlloc,
      InternalServer::ResponseRelease, nullptr /* StartFn*/));
  THROW_IF_TRITON_ERR(TRITONSERVER_ResponseAllocatorSetQueryFunction(
      allocator_, OutputBufferQuery));

  // Initialize trace manager
  if (options.trace_) {
    trace_manager_ = std::make_shared<TraceManager>(
        ToTritonTraceLevel(options.trace_->level_), options.trace_->rate_,
        options.trace_->count_, options.trace_->log_frequency_,
        options.trace_->file_);
  } else {
    // Tracing is not enabled.
    trace_manager_ = nullptr;
  }

  StartRepoPollThread();
}

InternalServer::~InternalServer()
{
  if (allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "Failed to delete allocator.");
  }

  StopRepoPollThread();
}

void
InternalServer::StartRepoPollThread()
{
  repo_poll_thread_ = std::thread([this]() {
    while (!is_exiting_) {
      if (repository_poll_secs_ > 0) {
        THROW_IF_TRITON_ERR(
            TRITONSERVER_ServerPollModelRepository(server_.get()));
      }
      std::unique_lock<std::mutex> lock(exit_mu_);
      std::chrono::seconds wait_timeout(
          (repository_poll_secs_ == 0) ? 3600 : repository_poll_secs_);
      exit_cv_.wait_for(lock, wait_timeout);
    }
  });
}

void
InternalServer::StopRepoPollThread()
{
  std::unique_lock<std::mutex> lock(exit_mu_);
  is_exiting_ = true;
  exit_cv_.notify_all();
  repo_poll_thread_.detach();
}

std::future<std::unique_ptr<InferResult>>
InternalServer::AsyncInfer(InferRequest& infer_request)
{
  std::future<std::unique_ptr<InferResult>> result_future;
  // The inference request object for sending internal requests.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  try {
    uint32_t txn_flags;
    THROW_IF_TRITON_ERR(TRITONSERVER_ServerModelTransactionProperties(
        server_.get(), infer_request.infer_options_->model_name_.c_str(),
        infer_request.infer_options_->model_version_, &txn_flags,
        nullptr /* voidp */));
    infer_request.is_decoupled_ =
        ((txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0);

    AsyncInferHelper(&irequest, infer_request);

    TRITONSERVER_InferenceTrace* triton_trace = nullptr;
    if (trace_manager_) {
      // Update trace setting for specified model if needed.
      if (infer_request.infer_options_->trace_) {
        TraceManager::TraceSetting new_setting(
            ToTritonTraceLevel(infer_request.infer_options_->trace_->level_),
            infer_request.infer_options_->trace_->rate_,
            infer_request.infer_options_->trace_->count_,
            infer_request.infer_options_->trace_->log_frequency_,
            std::make_shared<TraceManager::TraceFile>(
                infer_request.infer_options_->trace_->file_));
        trace_manager_->UpdateTraceSetting(
            infer_request.infer_options_->model_name_, new_setting);
      }
      infer_request.trace_ = std::move(trace_manager_->SampleTrace(
          infer_request.infer_options_->model_name_));
      if (infer_request.trace_ != nullptr) {
        triton_trace = infer_request.trace_->trace_;
      }
    } else if (infer_request.infer_options_->trace_) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("error when updating trace setting for model '") +
           infer_request.infer_options_->model_name_ +
           "': tracing is not enabled.")
              .c_str());
    }

    {
      auto p = new std::promise<std::unique_ptr<InferResult>>();
      result_future = p->get_future();
      infer_request.prev_promise_.reset(std::move(p));

      if (infer_request.infer_options_->custom_allocator_ == nullptr) {
        THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator_, reinterpret_cast<void*>(&infer_request),
            InternalServer::InferResponseComplete,
            reinterpret_cast<void*>(&infer_request)));
      } else {
        THROW_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, InternalRequest::custom_triton_allocator_,
            nullptr /* response_allocator_userp */,
            InternalServer::InferResponseComplete,
            reinterpret_cast<void*>(&infer_request)));
      }
      THROW_IF_TRITON_ERR(
          TRITONSERVER_ServerInferAsync(server_.get(), irequest, triton_trace));
    }
  }
  catch (const TritonException& ex) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(irequest),
        "Failed to delete inference request.");
    throw TritonException(std::string("Error - AsyncInfer: ") + ex.what());
  }

  return result_future;
}

std::unique_ptr<GenericInferRequest> 
GenericInferRequest::Create(const InferOptions& options) {
  return InferRequest::Create(options);
}
GenericInferRequest::~GenericInferRequest() {}

std::unique_ptr<InferRequest>
InferRequest::Create(const InferOptions& options)
{
  std::unique_ptr<InternalRequest> internal_request;
  internal_request.reset(new InternalRequest(options));
  return internal_request;
}

InferRequest::InferRequest() : is_decoupled_(false)
{
  str_bufs_.clear();
  inputs_.clear();
  outputs_.clear();
}

InferRequest::~InferRequest() {}

InternalRequest::InternalRequest(const InferOptions& options) : InferRequest()
{
  infer_options_.reset(new InferOptions(
      options.model_name_, options.model_version_, options.request_id_,
      options.correlation_id_, options.correlation_id_str_,
      options.sequence_start_, options.sequence_end_, options.priority_,
      options.request_timeout_, options.custom_allocator_, options.trace_));

  // Store custom allocator as a static variable as it's needed in global
  // functions.
  custom_allocator_ = options.custom_allocator_;
  custom_triton_allocator_ = nullptr;
  // Initialize custom allocator if it's set.
  if (options.custom_allocator_ != nullptr) {
    THROW_IF_TRITON_ERR(TRITONSERVER_ResponseAllocatorNew(
        &custom_triton_allocator_, CustomAllocFn,
        InternalServer::ResponseRelease, CustomStartFn));
    THROW_IF_TRITON_ERR(TRITONSERVER_ResponseAllocatorSetQueryFunction(
        custom_triton_allocator_, OutputBufferQuery));
  }
}

InternalRequest::~InternalRequest()
{
  if (custom_triton_allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(custom_triton_allocator_),
        "Failed to delete allocator.");
    custom_triton_allocator_ = nullptr;
  }
}

void
InferRequest::AddInput(
    const std::string& name, const Tensor& input_tensor) noexcept
{
  inputs_[name] = std::make_unique<Tensor>(input_tensor);
}

void
InferRequest::AddRequestedOutput(const std::string& name, Tensor& output_tensor)
{
  try {
    if (output_tensor.buffer_ == nullptr) {
      throw TritonException(
          "Pre-allocated buffer for '" + name + "' is a nullptr.");
    }
    std::unique_ptr<InferRequestedOutput> output = InferRequestedOutput::Create(
        name, output_tensor.buffer_, output_tensor.byte_size_,
        output_tensor.memory_type_, output_tensor.memory_type_id_);
    outputs_.push_back(std::move(output));
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - AddRequestedOutput: ") + ex.what());
  }
}

void
InferRequest::AddRequestedOutput(const std::string& name)
{
  try {
    std::unique_ptr<InferRequestedOutput> output =
        InferRequestedOutput::Create(name);
    outputs_.push_back(std::move(output));
  }
  catch (const TritonException& ex) {
    throw TritonException(
        std::string("Error - AddRequestedOutput: ") + ex.what());
  }
}

void
InferRequest::Reset()
{
  inputs_.clear();
  outputs_.clear();
  tensor_alloc_map_.clear();
}

GenericInferResult::~GenericInferResult() {}

InferResult::InferResult()
    : has_error_(false), error_msg_(""), completed_response_(nullptr)
{
}

InferResult::~InferResult() {}

InternalResult::InternalResult() : InferResult() {}

InternalResult::~InternalResult()
{
  if (completed_response_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceResponseDelete(completed_response_),
        "Failed to delete inference response.");
  }
}

void
InternalResult::FinalizeResponse(
    TRITONSERVER_InferenceResponse* response, const AllocInfo& alloc_info)
{
  try {
    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceResponseError(response));

    const char* model_name;
    int64_t model_version;
    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceResponseModel(
        response, &model_name, &model_version));
    const char* request_id = nullptr;
    THROW_IF_TRITON_ERR(
        TRITONSERVER_InferenceResponseId(response, &request_id));
    model_name_ = model_name;
    model_version_ = model_version;
    request_id_ = request_id;

    uint32_t parameter_count;
    THROW_IF_TRITON_ERR(TRITONSERVER_InferenceResponseParameterCount(
        response, &parameter_count));
    for (uint32_t pidx = 0; pidx < parameter_count; ++pidx) {
      const char* name;
      TRITONSERVER_ParameterType type;
      const void* vvalue;
      THROW_IF_TRITON_ERR(TRITONSERVER_InferenceResponseParameter(
          response, pidx, &name, &type, &vvalue));
      params_.push_back(std::move(std::unique_ptr<ResponseParameters>(
          new ResponseParameters(name, type, vvalue))));
    }

    uint32_t output_count;
    THROW_IF_TRITON_ERR(
        TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

    for (uint32_t idx = 0; idx < output_count; ++idx) {
      const char* cname;
      TRITONSERVER_DataType datatype;
      const int64_t* shape;
      uint64_t dim_count;
      const void* base;
      size_t byte_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      void* userp;
      THROW_IF_TRITON_ERR(TRITONSERVER_InferenceResponseOutput(
          response, idx, &cname, &datatype, &shape, &dim_count, &base,
          &byte_size, &memory_type, &memory_type_id, &userp));

      std::string name(cname);

      std::vector<int64_t> output_shape;
      for (uint64_t i = 0; i < dim_count; i++) {
        output_shape.push_back(*(shape + i));
      }

      MemoryType mem_type = TritonToMemoryType(memory_type);
      // FIXME (DLIS-4134) Need to investigate further for the performance of
      // using unordered_map vs. ordered vector with access via
      // std::lower_bound.
      infer_outputs_[name] = std::make_shared<Tensor>(
          const_cast<char*>(reinterpret_cast<const char*>(base)), byte_size,
          TritonToDataType(datatype), output_shape, mem_type, memory_type_id);

      // Set allocation info for the output tensor.
      infer_outputs_[name]->custom_allocator_ = alloc_info.first;
      if (alloc_info.second.find(name) != alloc_info.second.end()) {
        infer_outputs_[name]->is_pre_alloc_ = true;
      }
      infer_outputs_[name]->is_output_ = true;
    }
  }
  catch (const TritonException& ex) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_InferenceResponseDelete(response),
          "Failed to delete inference response.");
      response = nullptr;
    }
    has_error_ = true;
    error_msg_ = ex.what();
  }
  // Store the completed response to InferResult.
  completed_response_ = response;
}

std::string
InferResult::ModelName() noexcept
{
  return model_name_;
}

std::string
InferResult::ModelVersion() noexcept
{
  return std::to_string(model_version_);
}

std::string
InferResult::Id() noexcept
{
  return request_id_;
}

std::vector<std::string>
InferResult::OutputNames()
{
  std::vector<std::string> output_names;
  for (const auto& outputs : infer_outputs_) {
    output_names.push_back(outputs.first);
  }

  return output_names;
}

std::shared_ptr<Tensor>
InferResult::Output(const std::string& name)
{
  std::shared_ptr<Tensor> output;
  if (infer_outputs_.find(name) != infer_outputs_.end()) {
    output = infer_outputs_[name];
  } else {
    throw TritonException(
        std::string("Error - Output: ") +
        "The response does not contain result for output '" + name + "'.");
  }

  return output;
}

std::vector<std::string>
InferResult::StringData(const std::string& name)
{
  std::vector<std::string> string_result;
  if (infer_outputs_.find(name) != infer_outputs_.end()) {
    if (infer_outputs_[name]->data_type_ == DataType::BYTES) {
      const char* buf =
          reinterpret_cast<const char*>(infer_outputs_[name]->buffer_);
      size_t byte_size = infer_outputs_[name]->byte_size_;

      string_result.clear();
      size_t buf_offset = 0;
      while (byte_size > buf_offset) {
        const uint32_t element_size =
            *(reinterpret_cast<const char*>(buf + buf_offset));
        string_result.emplace_back(
            (buf + buf_offset + sizeof(element_size)), element_size);
        buf_offset += (sizeof(element_size) + element_size);
      }
    } else {
      throw TritonException(
          std::string("Error - StringData: ") +
          "The data type of the output '" + name + "' is not 'BYTES.");
    }
  } else {
    throw TritonException(
        std::string("Error - StringData: ") +
        "The response does not contain result for output '" + name + "'.");
  }

  return string_result;
}

std::string
InferResult::DebugString()
{
  try {
    triton::common::TritonJson::Value response_json(
        triton::common::TritonJson::ValueType::OBJECT);
    if ((request_id_ != nullptr) && (request_id_[0] != '\0')) {
      THROW_IF_TRITON_ERR(response_json.AddStringRef("id", request_id_));
    }
    THROW_IF_TRITON_ERR(response_json.AddStringRef("model_name", model_name_));
    THROW_IF_TRITON_ERR(response_json.AddString(
        "model_version", std::move(std::to_string(model_version_))));

    if (!params_.empty()) {
      triton::common::TritonJson::Value params_json(
          response_json, triton::common::TritonJson::ValueType::OBJECT);
      for (size_t i = 0; i < params_.size(); i++) {
        switch (params_[i]->type_) {
          case TRITONSERVER_PARAMETER_BOOL:
            THROW_IF_TRITON_ERR(params_json.AddBool(
                params_[i]->name_,
                *(reinterpret_cast<const bool*>(params_[i]->vvalue_))));
            break;
          case TRITONSERVER_PARAMETER_INT:
            THROW_IF_TRITON_ERR(params_json.AddInt(
                params_[i]->name_,
                *(reinterpret_cast<const int64_t*>(params_[i]->vvalue_))));
            break;
          case TRITONSERVER_PARAMETER_STRING:
            THROW_IF_TRITON_ERR(params_json.AddStringRef(
                params_[i]->name_,
                reinterpret_cast<const char*>(params_[i]->vvalue_)));
            break;
          case TRITONSERVER_PARAMETER_BYTES:
            throw TritonException(
                std::string("Error - DebugString: ") +
                "Response parameter of type 'TRITONSERVER_PARAMETER_BYTES' is "
                "not currently supported");
        }
      }
      THROW_IF_TRITON_ERR(
          response_json.Add("parameters", std::move(params_json)));
    }

    triton::common::TritonJson::Value response_outputs(
        response_json, triton::common::TritonJson::ValueType::ARRAY);
    for (auto& infer_output : infer_outputs_) {
      std::shared_ptr<Tensor> output = infer_output.second;
      triton::common::TritonJson::Value output_json(
          response_json, triton::common::TritonJson::ValueType::OBJECT);
      THROW_IF_TRITON_ERR(
          output_json.AddStringRef("name", infer_output.first.c_str()));
      const char* datatype_str = DataTypeString(output->data_type_).c_str();
      THROW_IF_TRITON_ERR(output_json.AddStringRef("datatype", datatype_str));
      triton::common::TritonJson::Value shape_json(
          response_json, triton::common::TritonJson::ValueType::ARRAY);
      for (size_t j = 0; j < output->shape_.size(); j++) {
        THROW_IF_TRITON_ERR(shape_json.AppendUInt(output->shape_[j]));
      }
      THROW_IF_TRITON_ERR(output_json.Add("shape", std::move(shape_json)));
      THROW_IF_TRITON_ERR(response_outputs.Append(std::move(output_json)));
    }
    THROW_IF_TRITON_ERR(
        response_json.Add("outputs", std::move(response_outputs)));


    triton::common::TritonJson::WriteBuffer buffer;
    THROW_IF_TRITON_ERR(response_json.Write(&buffer));
    return buffer.Contents();
  }
  catch (const TritonException& ex) {
    throw TritonException(std::string("Error - DebugString: ") + ex.what());
  }
}

bool
InferResult::HasError()
{
  return has_error_;
}

std::string
InferResult::ErrorMsg()
{
  return error_msg_;
}

std::unique_ptr<std::future<std::unique_ptr<InferResult>>>
InferResult::GetNextResult()
{
  return std::move(next_result_future_);
}

}}}  // namespace triton::developer_tools::server
