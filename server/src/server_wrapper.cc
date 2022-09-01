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
#include <sstream>
#include <string>
#include <vector>
#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

namespace triton { namespace server { namespace wrapper {

using AllocInfo = std::pair<std::shared_ptr<Allocator>, TensorAllocMap>;

//==============================================================================
/// Helper functions
///
std::string
WrapperDataTypeString(const DataType& data_type)
{
  switch (data_type) {
    case BOOL:
      return "BOOL";
    case UINT8:
      return "UINT8";
    case UINT16:
      return "UINT16";
    case UINT32:
      return "UINT32";
    case UINT64:
      return "UINT64";
    case INT8:
      return "INT8";
    case INT16:
      return "INT16";
    case INT32:
      return "INT32";
    case INT64:
      return "INT64";
    case FP16:
      return "FP16";
    case FP32:
      return "FP32";
    case FP64:
      return "FP64";
    case BYTES:
      return "BYTES";
    case BF16:
      return "BF16";
    default:
      break;
  }

  return "<invalid>";
}

std::string
WrapperMemoryTypeString(const MemoryType& memory_type)
{
  switch (memory_type) {
    case CPU:
      return "CPU";
    case CPU_PINNED:
      return "CPU_PINNED";
    case GPU:
      return "GPU";
    default:
      break;
  }

  return "<invalid>";
}

std::string
WrapperModelReadyStateString(const ModelReadyState& state)
{
  switch (state) {
    case UNKNOWN:
      return "UNKNOWN";
    case READY:
      return "READY";
    case UNAVAILABLE:
      return "UNAVAILABLE";
    case LOADING:
      return "LOADING";
    case UNLOADING:
      return "UNLOADING";
    default:
      return "UNKNOWN";
  }
}
//==============================================================================
/// InternalServer class
///
class InternalServer : public TritonServer {
 public:
  InternalServer(const ServerOptions& server_options);

  ~InternalServer();

  // Callback functions for pre-allocated output buffer.
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

  Error AsyncInfer(
      std::future<std::unique_ptr<InferResult>>* result_future,
      const InferRequest& infer_request) override;

  // The allocator object allocating output tensor.
  static TRITONSERVER_ResponseAllocator* allocator_;
};

TRITONSERVER_ResponseAllocator* InternalServer::allocator_;

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
    RETURN_TRITON_ERR_IF_ERR(
        InternalRequest::custom_allocator_->StartFn()(userp));
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
    MemoryType preferred_mem_type;
    MemoryType actual_mem_type;
    RETURN_TRITON_ERR_IF_ERR(
        TritonToWrapperMemoryType(&preferred_mem_type, preferred_memory_type));
    RETURN_TRITON_ERR_IF_ERR(
        TritonToWrapperMemoryType(&actual_mem_type, *actual_memory_type));

    RETURN_TRITON_ERR_IF_ERR(InternalRequest::custom_allocator_->AllocFn()(
        tensor_name, byte_size, preferred_mem_type, preferred_memory_type_id,
        buffer, &actual_mem_type, actual_memory_type_id));

    RETURN_TRITON_ERR_IF_ERR(
        WrapperToTritonMemoryType(actual_memory_type, actual_mem_type));
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Custom response allocation function is not set.");
  }
  *buffer_userp = nullptr;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if (request != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "Failed to delete inference request.");
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    auto p = reinterpret_cast<
        std::pair<std::promise<std::unique_ptr<InferResult>>*, AllocInfo>*>(
        userp);
    std::unique_ptr<InternalResult> result = std::make_unique<InternalResult>();
    result->FinalizeResponse(response, p->second);
    std::unique_ptr<InferResult> infer_result = std::move(result);
    p->first->set_value(std::move(infer_result));
    delete p->first;
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
{
  verbose_ = 0;
  info_ = true;
  warn_ = true;
  error_ = true;
  format_ = LOG_DEFAULT;
  log_file_ = "";
}

LoggingOptions::LoggingOptions(
    const uint verbose, const bool info, const bool warn, const bool error,
    const LogFormat& format, const std::string& log_file)
{
  verbose_ = verbose;
  info_ = info;
  warn_ = warn;
  error_ = error;
  format_ = format;
  log_file_ = log_file;
}

MetricsOptions::MetricsOptions()
{
  allow_metrics_ = true;
  allow_gpu_metrics_ = true;
  allow_cpu_metrics_ = true;
  metrics_interval_ms_ = 2000;
}

MetricsOptions::MetricsOptions(
    const bool allow_metrics, const bool allow_gpu_metrics,
    const bool allow_cpu_metrics, const uint64_t metrics_interval_ms)
{
  allow_metrics_ = allow_metrics;
  allow_gpu_metrics_ = allow_gpu_metrics;
  allow_cpu_metrics_ = allow_cpu_metrics;
  metrics_interval_ms_ = metrics_interval_ms;
}

BackendConfig::BackendConfig()
{
  backend_name_ = "";
  setting_ = "";
  value_ = "";
}

BackendConfig::BackendConfig(
    const std::string& backend_name, const std::string& setting,
    const std::string& value)
{
  backend_name_ = backend_name;
  setting_ = setting;
  value_ = value;
}

ServerOptions::ServerOptions(
    const std::vector<std::string>& model_repository_paths)
{
  model_repository_paths_ = model_repository_paths;
  logging_ = LoggingOptions();
  metrics_ = MetricsOptions();
  be_config_.clear();
  server_id_ = "triton";
  backend_dir_ = "/opt/tritonserver/backends";
  repo_agent_dir_ = "/opt/tritonserver/repoagents";
  disable_auto_complete_config_ = false;
  model_control_mode_ = MODEL_CONTROL_NONE;
}

ServerOptions::ServerOptions(
    const std::vector<std::string>& model_repository_paths,
    const LoggingOptions& logging, const MetricsOptions& metrics,
    const std::vector<BackendConfig>& be_config, const std::string& server_id,
    const std::string& backend_dir, const std::string& repo_agent_dir,
    const bool disable_auto_complete_config,
    const ModelControlMode& model_control_mode)
{
  model_repository_paths_ = model_repository_paths;
  logging_ = logging;
  metrics_ = metrics;
  be_config_ = be_config;
  server_id_ = server_id;
  backend_dir_ = backend_dir;
  repo_agent_dir_ = repo_agent_dir;
  disable_auto_complete_config_ = disable_auto_complete_config;
  model_control_mode_ = model_control_mode;
}

RepositoryIndex::RepositoryIndex(
    const std::string& name, const std::string& version,
    const ModelReadyState& state)
{
  name_ = name;
  version_ = version;
  state_ = state;
}

Tensor::Tensor(
    char* buffer, const size_t& byte_size, DataType data_type,
    std::vector<int64_t> shape, MemoryType memory_type, int64_t memory_type_id)
{
  buffer_ = buffer;
  byte_size_ = byte_size;
  data_type_ = data_type;
  shape_ = shape;
  memory_type_ = memory_type;
  memory_type_id_ = memory_type_id;
  custom_allocator_.reset();
  is_pre_alloc_ = false;
  is_output_ = false;
}

Tensor::Tensor(
    char* buffer, size_t byte_size, MemoryType memory_type,
    int64_t memory_type_id)
{
  buffer_ = buffer;
  byte_size_ = byte_size;
  data_type_ = INVALID;
  shape_ = {};
  memory_type_ = memory_type;
  memory_type_id_ = memory_type_id;
  custom_allocator_.reset();
  is_pre_alloc_ = false;
  is_output_ = false;
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
           std::to_string(byte_size_) + " in " +
           WrapperMemoryTypeString(memory_type_))
              .c_str());

      switch (memory_type_) {
        case CPU:
          free(buffer_);
          break;
#ifdef TRITON_ENABLE_GPU
        case CPU_PINNED: {
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
        case GPU: {
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
        auto err = custom_allocator_->ReleaseFn()(
            reinterpret_cast<void*>(buffer_), byte_size_, memory_type_,
            memory_type_id_);
        if (!err.IsOk()) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              ("error: using custom allocator - " + err.Message()).c_str());
        }
      }
    }
  }
}

InferOptions::InferOptions(const std::string& model_name)
{
  model_name_ = model_name;
  model_version_ = -1;
  request_id_ = "";
  correlation_id_ = 0;
  correlation_id_str_ = "";
  sequence_start_ = false;
  sequence_end_ = false;
  priority_ = 0;
  request_timeout_ = 0;
  custom_allocator_ = nullptr;
}

InferOptions::InferOptions(
    const std::string& model_name, const int64_t& model_version,
    const std::string& request_id, const uint64_t& correlation_id,
    const std::string& correlation_id_str, const bool sequence_start,
    const bool sequence_end, const uint64_t& priority,
    const uint64_t& request_timeout,
    std::shared_ptr<Allocator> custom_allocator)
{
  model_name_ = model_name;
  model_version_ = model_version;
  request_id_ = request_id;
  correlation_id_ = correlation_id;
  correlation_id_str_ = correlation_id_str;
  sequence_start_ = sequence_start;
  sequence_end_ = sequence_end;
  priority_ = priority;
  request_timeout_ = request_timeout;
  custom_allocator_ = custom_allocator;
}

std::unique_ptr<TritonServer>
TritonServer::Create(const ServerOptions& options)
{
  std::unique_ptr<InternalServer> internal_server;
  internal_server.reset(new InternalServer(options));
  return internal_server;
}

TritonServer::~TritonServer() {}

Error
TritonServer::LoadModel(const std::string& model_name)
{
  RETURN_ERR_IF_TRITON_ERR(
      TRITONSERVER_ServerLoadModel(server_.get(), model_name.c_str()));

  return Error::Success;
}

Error
TritonServer::UnloadModel(const std::string& model_name)
{
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_ServerUnloadModelAndDependents(
      server_.get(), model_name.c_str()));

  return Error::Success;
}

Error
TritonServer::LoadedModels(std::set<std::string>* loaded_models)
{
  std::vector<RepositoryIndex> repository_index;
  RETURN_IF_ERR(ModelIndex(&repository_index));

  std::set<std::string> models;
  for (size_t i = 0; i < repository_index.size(); i++) {
    models.insert(repository_index[i].name_);
  }

  *loaded_models = models;
  return Error::Success;
}

Error
TritonServer::ModelIndex(std::vector<RepositoryIndex>* repository_index)
{
  TRITONSERVER_Message* message = nullptr;
  uint32_t flags = TRITONSERVER_INDEX_FLAG_READY;
  RETURN_ERR_IF_TRITON_ERR(
      TRITONSERVER_ServerModelIndex(server_.get(), flags, &message));
  const char* buffer;
  size_t byte_size;
  RETURN_ERR_IF_TRITON_ERR(
      TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size));

  common::TritonJson::Value repo_index;
  RETURN_ERR_IF_TRITON_ERR(repo_index.Parse(buffer, byte_size));
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_MessageDelete(message));

  for (size_t i = 0; i < repo_index.ArraySize(); i++) {
    triton::common::TritonJson::Value index;
    RETURN_ERR_IF_TRITON_ERR(repo_index.IndexAsObject(i, &index));
    std::string name, version, state;
    RETURN_ERR_IF_TRITON_ERR(index.MemberAsString("name", &name));
    RETURN_ERR_IF_TRITON_ERR(index.MemberAsString("version", &version));
    RETURN_ERR_IF_TRITON_ERR(index.MemberAsString("state", &state));
    repository_index->push_back(
        RepositoryIndex(name, version, StringToWrapperModelReadyState(state)));
  }

  return Error::Success;
}

Error
TritonServer::Metrics(std::string* metrics_str)
{
  TRITONSERVER_Metrics* metrics = nullptr;
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_ServerMetrics(server_.get(), &metrics));
  const char* base;
  size_t byte_size;
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_MetricsFormatted(
      metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size));
  *metrics_str = std::string(base, byte_size);
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_MetricsDelete(metrics));

  return Error::Success;
}

Error
TritonServer::PrepareInferenceRequest(
    TRITONSERVER_InferenceRequest** irequest, const InferRequest& request)
{
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestNew(
      irequest, server_.get(), request.infer_options_->model_name_.c_str(),
      request.infer_options_->model_version_));

  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetId(
      *irequest, request.infer_options_->request_id_.c_str()));
  if (request.infer_options_->correlation_id_str_.empty()) {
    RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetCorrelationId(
        *irequest, request.infer_options_->correlation_id_));
  } else {
    RETURN_ERR_IF_TRITON_ERR(
        TRITONSERVER_InferenceRequestSetCorrelationIdString(
            *irequest, request.infer_options_->correlation_id_str_.c_str()));
  }

  uint32_t flags = 0;
  if (request.infer_options_->sequence_start_) {
    flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
  }
  if (request.infer_options_->sequence_end_) {
    flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
  }
  RETURN_ERR_IF_TRITON_ERR(
      TRITONSERVER_InferenceRequestSetFlags(*irequest, flags));

  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetPriority(
      *irequest, request.infer_options_->priority_));

  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
      *irequest, request.infer_options_->request_timeout_));
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      *irequest, InferRequestComplete, nullptr /* request_release_userp */));

  return Error::Success;
}

Error
TritonServer::PrepareInferenceInput(
    TRITONSERVER_InferenceRequest* irequest, const InferRequest& request)
{
  for (auto& input : request.inputs_) {
    RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestAddInput(
        irequest, input.first.c_str(),
        WrapperToTritonDataType(input.second->data_type_),
        input.second->shape_.data(), input.second->shape_.size()));

    TRITONSERVER_MemoryType memory_type;
    RETURN_IF_ERR(
        WrapperToTritonMemoryType(&memory_type, input.second->memory_type_));
    RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_InferenceRequestAppendInputData(
        irequest, input.first.c_str(), input.second->buffer_,
        input.second->byte_size_, memory_type, input.second->memory_type_id_));
  }

  return Error::Success;
}

Error
TritonServer::PrepareInferenceOutput(
    TRITONSERVER_InferenceRequest* irequest, InferRequest& request)
{
  for (auto& infer_output : request.outputs_) {
    const char* name = infer_output->Name().c_str();
    RETURN_ERR_IF_TRITON_ERR(
        TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, name));
    if (infer_output->Buffer() != nullptr) {
      request.tensor_alloc_map_[name] = std::make_tuple(
          infer_output->Buffer(), infer_output->ByteSize(),
          infer_output->MemoryType(), infer_output->MemoryTypeId());
    }
  }

  return Error::Success;
}

Error
TritonServer::AsyncInferHelper(
    TRITONSERVER_InferenceRequest** irequest, const InferRequest& infer_request)
{
  bool is_ready = false;
  std::string model_name = infer_request.infer_options_->model_name_.c_str();
  RETURN_ERR_IF_TRITON_ERR(TRITONSERVER_ServerModelIsReady(
      server_.get(), model_name.c_str(),
      infer_request.infer_options_->model_version_, &is_ready));

  if (!is_ready) {
    return Error(
        (std::string("Failed for execute the inference request. Model '") +
         model_name + "' is not ready.")
            .c_str());
  }

  RETURN_IF_ERR(PrepareInferenceRequest(irequest, infer_request));
  RETURN_IF_ERR(PrepareInferenceInput(*irequest, infer_request));
  RETURN_IF_ERR(PrepareInferenceOutput(
      *irequest, const_cast<InferRequest&>(infer_request)));

  return Error::Success;
}

InternalServer::InternalServer(const ServerOptions& options)
{
  uint32_t api_version_major, api_version_minor;
  THROW_IF_TRITON_ERR(
      TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor));
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    throw ServerWrapperException("triton server API version mismatch");
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
      server_options, options.logging_.verbose_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogInfo(
      server_options, options.logging_.info_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogWarn(
      server_options, options.logging_.warn_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetLogError(
      server_options, options.logging_.error_));
  TRITONSERVER_LogFormat log_format;
  THROW_IF_ERR(WrapperToTritonLogFormat(&log_format, options.logging_.format_));
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
        server_options, bc.backend_name_.c_str(), bc.setting_.c_str(),
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
  TRITONSERVER_ModelControlMode model_control_mode;
  THROW_IF_ERR(WrapperToTritonModelControlMode(
      &model_control_mode, options.model_control_mode_));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsSetModelControlMode(
      server_options, model_control_mode));

  TRITONSERVER_Server* server_ptr;
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options));
  THROW_IF_TRITON_ERR(TRITONSERVER_ServerOptionsDelete(server_options));
  server_ = std::shared_ptr<TRITONSERVER_Server>(
      server_ptr, TRITONSERVER_ServerDelete);

  // Initialize allocator
  allocator_ = nullptr;
  THROW_IF_TRITON_ERR(TRITONSERVER_ResponseAllocatorNew(
      &InternalServer::allocator_, InternalServer::ResponseAlloc,
      InternalServer::ResponseRelease, nullptr /* StartFn*/));
  THROW_IF_TRITON_ERR(TRITONSERVER_ResponseAllocatorSetQueryFunction(
      InternalServer::allocator_, OutputBufferQuery));
}

InternalServer::~InternalServer()
{
  if (allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorDelete(allocator_),
        "Failed to delete allocator.");
  }
}

Error
InternalServer::AsyncInfer(
    std::future<std::unique_ptr<InferResult>>* result_future,
    const InferRequest& infer_request)
{
  // The inference request object for sending internal requests.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  try {
    THROW_IF_ERR(AsyncInferHelper(&irequest, infer_request));
    {
      auto p = new std::promise<std::unique_ptr<InferResult>>();
      *result_future = p->get_future();

      // Construct the allocation info of an output tensor and pass it to the
      // callback function so that we can store the information in the ouput
      // 'Tensor' object. When calling the destructor of an output tensor, it
      // will know how to clean the buffer correctly.
      AllocInfo alloc_info(
          infer_request.infer_options_->custom_allocator_,
          infer_request.tensor_alloc_map_);
      auto result =
          new std::pair<std::promise<std::unique_ptr<InferResult>>*, AllocInfo>(
              p, alloc_info);

      if (infer_request.infer_options_->custom_allocator_ == nullptr) {
        THROW_ERR_IF_TRITON_ERR(
            TRITONSERVER_InferenceRequestSetResponseCallback(
                irequest, allocator_,
                reinterpret_cast<void*>(
                    const_cast<InferRequest*>(&infer_request)),
                InferResponseComplete, reinterpret_cast<void*>(result)));
      } else {
        THROW_ERR_IF_TRITON_ERR(
            TRITONSERVER_InferenceRequestSetResponseCallback(
                irequest, InternalRequest::custom_triton_allocator_,
                nullptr /* response_allocator_userp */, InferResponseComplete,
                reinterpret_cast<void*>(result)));
      }

      THROW_ERR_IF_TRITON_ERR(TRITONSERVER_ServerInferAsync(
          server_.get(), irequest, nullptr /* trace */));
    }
  }
  catch (const ServerWrapperException& ex) {
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(irequest),
        "Failed to delete inference request.");
    return Error(ex.what());
  }

  return Error::Success;
}

std::unique_ptr<InferRequest>
InferRequest::Create(const InferOptions& options)
{
  std::unique_ptr<InternalRequest> internal_request;
  internal_request.reset(new InternalRequest(options));
  return internal_request;
}

InferRequest::~InferRequest() {}

InternalRequest::InternalRequest(const InferOptions& options)
{
  infer_options_.reset(new InferOptions(
      options.model_name_, options.model_version_, options.request_id_,
      options.correlation_id_, options.correlation_id_str_,
      options.sequence_start_, options.sequence_end_, options.priority_,
      options.request_timeout_, options.custom_allocator_));

  // Store custom allocator as a static variable as it's needed in global
  // functions.
  custom_allocator_ = options.custom_allocator_;
  custom_triton_allocator_ = nullptr;
  // Initialize custom allocator if it's set.
  if (options.custom_allocator_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorNew(
            &custom_triton_allocator_, CustomAllocFn,
            InternalServer::ResponseRelease, CustomStartFn),
        "Creating custom allocator");
    LOG_IF_ERROR(
        TRITONSERVER_ResponseAllocatorSetQueryFunction(
            custom_triton_allocator_, OutputBufferQuery),
        "Setting query function for allocator");
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

Error
InferRequest::AddInput(const std::string& name, const Tensor& input_tensor)
{
  inputs_[name] = std::make_unique<Tensor>(input_tensor);

  return Error::Success;
}

Error
InferRequest::AddRequestedOutput(const std::string& name, Tensor& output_tensor)
{
  std::unique_ptr<InferRequestedOutput> output;
  if (output_tensor.buffer_ == nullptr) {
    return Error("Pre-allocated buffer for '" + name + "' is a nullptr.");
  } else {
    RETURN_IF_ERR(InferRequestedOutput::Create(
        output, name, output_tensor.buffer_, output_tensor.byte_size_,
        output_tensor.memory_type_, output_tensor.memory_type_id_));
  }

  outputs_.push_back(std::move(output));

  return Error::Success;
}

Error
InferRequest::AddRequestedOutput(const std::string& name)
{
  std::unique_ptr<InferRequestedOutput> output;
  output = InferRequestedOutput::Create(name);
  outputs_.push_back(std::move(output));

  return Error::Success;
}

Error
InferRequest::Reset()
{
  inputs_.clear();
  outputs_.clear();
  tensor_alloc_map_.clear();

  return Error::Success;
}

InferResult::~InferResult()
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

    std::unordered_map<std::string, std::vector<char>> output_data;
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

      MemoryType mem_type;
      THROW_IF_ERR(TritonToWrapperMemoryType(&mem_type, memory_type));
      infer_outputs_[name] = std::make_shared<Tensor>(
          const_cast<char*>(reinterpret_cast<const char*>(base)), byte_size,
          TritonToWrapperDataType(datatype), output_shape, mem_type,
          memory_type_id);

      // Set allocation info for the output tensor.
      infer_outputs_[name]->custom_allocator_ = alloc_info.first;
      if (alloc_info.second.find(name) != alloc_info.second.end()) {
        infer_outputs_[name]->is_pre_alloc_ = true;
      }
      infer_outputs_[name]->is_output_ = true;
    }
  }
  catch (const ServerWrapperException& ex) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_InferenceResponseDelete(response),
          "Failed to delete inference response.");
      response = nullptr;
    }
    response_error_ = Error(
        std::string("Error when finalizing the infer response: ") +
        std::string(ex.what()));
  }
  // Store the completed response to InferResult.
  completed_response_ = response;
}

bool
InferResult::HasError()
{
  return !response_error_.IsOk();
}

std::string
InferResult::ErrorMsg()
{
  return response_error_.Message();
}

std::string
InferResult::ModelName()
{
  return model_name_;
}

std::string
InferResult::ModelVersion()
{
  return std::to_string(model_version_);
}

std::string
InferResult::Id()
{
  return request_id_;
}

Error
InferResult::Output(const std::string& name, std::shared_ptr<Tensor>& output)
{
  if (!response_error_.IsOk()) {
    return Error(
        "Error when retrieving output '" + name +
        "'. An error occurs during inference. Use 'ErrorMsg()' for checking "
        "error message.");
  }

  if (infer_outputs_.find(name) != infer_outputs_.end()) {
    output = infer_outputs_[name];
  } else {
    return Error(
        "The response does not contain result for output '" + name + "'.");
  }

  return Error::Success;
}

Error
InferResult::StringData(
    const std::string& name, std::vector<std::string>* string_result)
{
  if (!response_error_.IsOk()) {
    return Error(
        "Error when retrieving StringData for '" + name +
        "'. An error occurs during inference. Use 'ErrorMsg()' for checking "
        "error message.");
  }

  if (infer_outputs_.find(name) != infer_outputs_.end()) {
    if (infer_outputs_[name]->data_type_ == BYTES) {
      const char* buf =
          reinterpret_cast<const char*>(infer_outputs_[name]->buffer_);
      size_t byte_size = infer_outputs_[name]->byte_size_;

      string_result->clear();
      size_t buf_offset = 0;
      while (byte_size > buf_offset) {
        const uint32_t element_size =
            *(reinterpret_cast<const char*>(buf + buf_offset));
        string_result->emplace_back(
            (buf + buf_offset + sizeof(element_size)), element_size);
        buf_offset += (sizeof(element_size) + element_size);
      }
    } else {
      return Error("The data type of the output '" + name + "' is not 'BYTES.");
    }
  } else {
    return Error(
        "The response does not contain result for output '" + name + "'.");
  }

  return Error::Success;
}

Error
InferResult::DebugString(std::string* string_result)
{
  if (!response_error_.IsOk()) {
    return Error(
        "Error when retrieving DebugString. An error occurs during inference. "
        "Use 'ErrorMsg()' for checking error message.");
  }

  triton::common::TritonJson::Value response_json(
      triton::common::TritonJson::ValueType::OBJECT);
  if ((request_id_ != nullptr) && (request_id_[0] != '\0')) {
    RETURN_ERR_IF_TRITON_ERR(response_json.AddStringRef("id", request_id_));
  }
  RETURN_ERR_IF_TRITON_ERR(
      response_json.AddStringRef("model_name", model_name_));
  RETURN_ERR_IF_TRITON_ERR(response_json.AddString(
      "model_version", std::move(std::to_string(model_version_))));

  if (!params_.empty()) {
    triton::common::TritonJson::Value params_json(
        response_json, triton::common::TritonJson::ValueType::OBJECT);
    for (size_t i = 0; i < params_.size(); i++) {
      switch (params_[i]->type_) {
        case TRITONSERVER_PARAMETER_BOOL:
          RETURN_ERR_IF_TRITON_ERR(params_json.AddBool(
              params_[i]->name_,
              *(reinterpret_cast<const bool*>(params_[i]->vvalue_))));
          break;
        case TRITONSERVER_PARAMETER_INT:
          RETURN_ERR_IF_TRITON_ERR(params_json.AddInt(
              params_[i]->name_,
              *(reinterpret_cast<const int64_t*>(params_[i]->vvalue_))));
          break;
        case TRITONSERVER_PARAMETER_STRING:
          RETURN_ERR_IF_TRITON_ERR(params_json.AddStringRef(
              params_[i]->name_,
              reinterpret_cast<const char*>(params_[i]->vvalue_)));
          break;
        case TRITONSERVER_PARAMETER_BYTES:
          return Error(
              "Response parameter of type 'TRITONSERVER_PARAMETER_BYTES' is "
              "not currently supported");
      }
    }
    RETURN_ERR_IF_TRITON_ERR(
        response_json.Add("parameters", std::move(params_json)));
  }

  triton::common::TritonJson::Value response_outputs(
      response_json, triton::common::TritonJson::ValueType::ARRAY);
  for (auto& infer_output : infer_outputs_) {
    std::shared_ptr<Tensor> output = infer_output.second;
    triton::common::TritonJson::Value output_json(
        response_json, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_ERR_IF_TRITON_ERR(
        output_json.AddStringRef("name", infer_output.first.c_str()));
    const char* datatype_str =
        WrapperDataTypeString(output->data_type_).c_str();
    RETURN_ERR_IF_TRITON_ERR(
        output_json.AddStringRef("datatype", datatype_str));
    triton::common::TritonJson::Value shape_json(
        response_json, triton::common::TritonJson::ValueType::ARRAY);
    for (size_t j = 0; j < output->shape_.size(); j++) {
      RETURN_ERR_IF_TRITON_ERR(shape_json.AppendUInt(output->shape_[j]));
    }
    RETURN_ERR_IF_TRITON_ERR(output_json.Add("shape", std::move(shape_json)));
    RETURN_ERR_IF_TRITON_ERR(response_outputs.Append(std::move(output_json)));
  }
  RETURN_ERR_IF_TRITON_ERR(
      response_json.Add("outputs", std::move(response_outputs)));


  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_ERR_IF_TRITON_ERR(response_json.Write(&buffer));
  *string_result = buffer.Contents();

  return Error::Success;
}

}}}  // namespace triton::server::wrapper
