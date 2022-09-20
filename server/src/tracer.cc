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

#include "tracer.h"

#include <stdlib.h>
#include <unordered_map>
#include "triton/common/logging.h"
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU
#include <iostream>

namespace triton { namespace developer_tools { namespace server {

#define IGNORE_ERROR(X)                   \
  do {                                    \
    TRITONSERVER_Error* ie_err__ = (X);   \
    if (ie_err__ != nullptr) {            \
      TRITONSERVER_ErrorDelete(ie_err__); \
    }                                     \
  } while (false)

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

TraceManager::TraceManager(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    const int32_t count, const uint32_t log_frequency,
    const std::string& filepath)
{
  std::shared_ptr<TraceFile> file(new TraceFile(filepath));
  global_setting_.reset(
      new TraceSetting(level, rate, count, log_frequency, file));
  trace_files_.emplace(filepath, file);
}

void
TraceManager::UpdateTraceSetting(
    const std::string& model_name, const TraceSetting& new_setting)
{
  std::shared_ptr<TraceSetting> setting(new TraceSetting(
      new_setting.level_, new_setting.rate_, new_setting.count_,
      new_setting.log_frequency_, new_setting.file_));
  if ((!setting->Valid()) &&
      (new_setting.level_ != TRITONSERVER_TRACE_LEVEL_DISABLED)) {
    throw TritonException(
        std::string("Attempting to set invalid trace setting: ") +
        setting->Reason());
  }

  std::lock_guard<std::mutex> r_lk(r_mu_);
  auto it = model_settings_.find(model_name);
  if (it != model_settings_.end()) {
    // Model update
    it->second = std::move(setting);
  } else {
    // Model init
    model_settings_.emplace(model_name, setting);
  }
}

std::shared_ptr<TraceManager::Trace>
TraceManager::SampleTrace(const std::string& model_name)
{
  std::shared_ptr<TraceSetting> trace_setting;
  {
    std::lock_guard<std::mutex> r_lk(r_mu_);
    auto m_it = model_settings_.find(model_name);
    trace_setting =
        (m_it == model_settings_.end()) ? global_setting_ : m_it->second;
  }
  std::shared_ptr<Trace> ts = trace_setting->SampleTrace();
  if (ts != nullptr) {
    ts->setting_ = trace_setting;
  }
  return ts;
}

void
TraceManager::TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp)
{
  uint64_t parent_id;
  LOG_IF_ERROR(
      TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
      "getting trace parent id");
  // The userp will be shared with the trace children, so only delete it
  // if the root trace is being released
  if (parent_id == 0) {
    delete reinterpret_cast<std::shared_ptr<TraceManager::Trace>*>(userp);
  }
  LOG_IF_ERROR(TRITONSERVER_InferenceTraceDelete(trace), "deleting trace");
}

void
TraceManager::TraceActivity(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    void* userp)
{
  uint64_t id;
  LOG_IF_ERROR(TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  // The function may be called with different traces but the same 'userp',
  // group the activity of the same trace together for more readable output.
  auto ts =
      reinterpret_cast<std::shared_ptr<TraceManager::Trace>*>(userp)->get();

  std::lock_guard<std::mutex> lk(ts->mtx_);
  std::stringstream* ss = nullptr;
  {
    if (ts->streams_.find(id) == ts->streams_.end()) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
      ss = ts->streams_[id].get();
      // If the string stream is not newly created, add "," as there is
      // already content in the string stream
      *ss << ",";
    }
  }

  // If 'activity' is TRITONSERVER_TRACE_REQUEST_START then collect
  // and serialize trace details.
  if (activity == TRITONSERVER_TRACE_REQUEST_START) {
    const char* model_name;
    int64_t model_version;
    uint64_t parent_id;

    LOG_IF_ERROR(
        TRITONSERVER_InferenceTraceModelName(trace, &model_name),
        "getting model name");
    LOG_IF_ERROR(
        TRITONSERVER_InferenceTraceModelVersion(trace, &model_version),
        "getting model version");
    LOG_IF_ERROR(
        TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
        "getting trace parent id");

    *ss << "{\"id\":" << id << ",\"model_name\":\"" << model_name
        << "\",\"model_version\":" << model_version;
    if (parent_id != 0) {
      *ss << ",\"parent_id\":" << parent_id;
    }
    *ss << "},";
  }

  *ss << "{\"id\":" << id << ",\"timestamps\":["
      << "{\"name\":\"" << TRITONSERVER_InferenceTraceActivityString(activity)
      << "\",\"ns\":" << timestamp_ns << "}]}";
}

void
TraceManager::TraceTensorActivity(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, const char* name,
    TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
    const int64_t* shape, uint64_t dim_count,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp)
{
  if ((activity != TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT) &&
      (activity != TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT) &&
      (activity != TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT)) {
    LOG_ERROR << "Unsupported activity: "
              << TRITONSERVER_InferenceTraceActivityString(activity);
    return;
  }

  void* buffer_base = const_cast<void*>(base);
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    buffer_base = malloc(byte_size);
    if (buffer_base == nullptr) {
      LOG_ERROR << "Failed to malloc CPU buffer";
      return;
    }
    cudaError_t err =
        cudaMemcpy(buffer_base, base, byte_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw TritonException(
          std::string("Error - copying buffer into CPU memory: ") +
          cudaGetErrorString(err));
    }

    // FAIL_IF_CUDA_ERR(
    //     cudaMemcpy(buffer_base, base, byte_size, cudaMemcpyDeviceToHost),
    //     "copying buffer into CPU memory");
#else
    LOG_ERROR << "GPU buffer is unsupported";
    return;
#endif  // TRITON_ENABLE_GPU
  }

  uint64_t id;
  LOG_IF_ERROR(TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  // The function may be called with different traces but the same 'userp',
  // group the activity of the same trace together for more readable output.
  auto ts =
      reinterpret_cast<std::shared_ptr<TraceManager::Trace>*>(userp)->get();

  std::lock_guard<std::mutex> lk(ts->mtx_);
  std::stringstream* ss = nullptr;
  {
    if (ts->streams_.find(id) == ts->streams_.end()) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
      ss = ts->streams_[id].get();
      // If the string stream is not newly created, add "," as there is
      // already content in the string stream
      *ss << ",";
    }
  }

  // collect and serialize trace details.
  *ss << "{\"id\":" << id << ",\"activity\":\""
      << TRITONSERVER_InferenceTraceActivityString(activity) << "\"";
  // collect tensor
  *ss << ",\"tensor\":{";
  // collect tensor name
  *ss << "\"name\":\"" << std::string(name) << "\"";
  // collect tensor data
  *ss << ",\"data\":\"";
  size_t element_count = 1;
  for (uint64_t i = 0; i < dim_count; i++) {
    element_count *= shape[i];
  }
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL: {
      const uint8_t* bool_base = reinterpret_cast<const uint8_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << ((bool_base[e] == 0) ? false : true);
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT8: {
      const uint8_t* cbase = reinterpret_cast<const uint8_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT16: {
      const uint16_t* cbase = reinterpret_cast<const uint16_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT32: {
      const uint32_t* cbase = reinterpret_cast<const uint32_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_UINT64: {
      const uint64_t* cbase = reinterpret_cast<const uint64_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT8: {
      const int8_t* cbase = reinterpret_cast<const int8_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT16: {
      const int16_t* cbase = reinterpret_cast<const int16_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT32: {
      const int32_t* cbase = reinterpret_cast<const int32_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INT64: {
      const int64_t* cbase = reinterpret_cast<const int64_t*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }

    // FP16 / BF16 already handled as binary blobs, no need to manipulate here
    case TRITONSERVER_TYPE_FP16: {
      break;
    }
    case TRITONSERVER_TYPE_BF16: {
      break;
    }

    case TRITONSERVER_TYPE_FP32: {
      const float* cbase = reinterpret_cast<const float*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_FP64: {
      const double* cbase = reinterpret_cast<const double*>(buffer_base);
      for (size_t e = 0; e < element_count; ++e) {
        *ss << cbase[e];
        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_BYTES: {
      const char* cbase = reinterpret_cast<const char*>(buffer_base);
      size_t offset = 0;
      for (size_t e = 0; e < element_count; ++e) {
        if ((offset + sizeof(uint32_t)) > byte_size) {
          return;
        }
        const size_t len = *(reinterpret_cast<const uint32_t*>(cbase + offset));
        offset += sizeof(uint32_t);
        if ((offset + len) > byte_size) {
          return;
        }
        std::string str(cbase + offset, len);
        *ss << "\\\"" << str << "\\\"";
        offset += len;

        if (e < (element_count - 1))
          *ss << ",";
      }
      break;
    }
    case TRITONSERVER_TYPE_INVALID: {
      return;
    }
  }
  *ss << "\",\"shape\":\"";
  for (uint64_t i = 0; i < dim_count; i++) {
    *ss << shape[i];
    if (i < (dim_count - 1)) {
      *ss << ",";
    }
  }
  *ss << "\",\"dtype\":\"" << TRITONSERVER_DataTypeString(datatype) << "\"}";
  *ss << "}";

  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (buffer_base != nullptr) {
      free(buffer_base);
    }
#endif  // TRITON_ENABLE_GPU
  }
}

TraceManager::Trace::~Trace()
{
  // Write trace now
  setting_->WriteTrace(streams_);
}

TraceManager::TraceFile::~TraceFile()
{
  if (!first_write_) {
    trace_file_ << "]";
  }
}

void
TraceManager::TraceFile::SaveTraces(
    std::stringstream& trace_stream, const bool to_index_file)
{
  try {
    if (to_index_file) {
      std::string file_name =
          file_name_ + "." + std::to_string(index_.fetch_add(1));
      std::ofstream file_stream;
      file_stream.open(file_name);
      file_stream << "[";
      file_stream << trace_stream.rdbuf();
      file_stream << "]";
    } else {
      std::lock_guard<std::mutex> lock(mu_);
      if (first_write_) {
        trace_file_.open(file_name_);
        trace_file_ << "[";
        first_write_ = false;
      } else {
        trace_file_ << ",";
      }
      trace_file_ << trace_stream.rdbuf();
    }
  }
  catch (const std::ofstream::failure& e) {
    LOG_ERROR << "failed creating trace file: " << e.what();
  }
  catch (...) {
    LOG_ERROR << "failed creating trace file: reason unknown";
  }
}

std::shared_ptr<TraceManager::Trace>
TraceManager::TraceSetting::SampleTrace()
{
  bool create_trace = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!Valid()) {
      return nullptr;
    }
    create_trace = (((++sample_) % rate_) == 0);
    if (create_trace && (count_ > 0)) {
      --count_;
      ++created_;
    }
  }
  if (create_trace) {
    std::shared_ptr<TraceManager::Trace> lts(new Trace());
    // Split 'Trace' management to frontend and Triton trace separately
    // to avoid dependency between frontend request and Triton trace's liveness
    auto trace_userp = new std::shared_ptr<TraceManager::Trace>(lts);
    TRITONSERVER_InferenceTrace* trace;
    TRITONSERVER_Error* err = TRITONSERVER_InferenceTraceTensorNew(
        &trace, level_, 0 /* parent_id */, TraceActivity, TraceTensorActivity,
        TraceRelease, trace_userp);
    if (err != nullptr) {
      LOG_IF_ERROR(err, "creating inference trace object");
      delete trace_userp;
      return nullptr;
    }
    lts->trace_ = trace;
    lts->trace_userp_ = trace_userp;
    LOG_IF_ERROR(
        TRITONSERVER_InferenceTraceId(trace, &lts->trace_id_),
        "getting trace id");
    return lts;
  }

  return nullptr;
}

void
TraceManager::TraceSetting::WriteTrace(
    const std::unordered_map<uint64_t, std::unique_ptr<std::stringstream>>&
        streams)
{
  std::unique_lock<std::mutex> lock(mu_);

  if (sample_in_stream_ != 0) {
    trace_stream_ << ",";
  }
  ++sample_in_stream_;
  ++collected_;

  size_t stream_count = 0;
  for (const auto& stream : streams) {
    trace_stream_ << stream.second->rdbuf();
    // Need to add ',' unless it is the last trace in the group
    ++stream_count;
    if (stream_count != streams.size()) {
      trace_stream_ << ",";
    }
  }
  // Write to file with index when one of the following is true
  // 1. trace_count is specified and that number of traces has been collected
  // 2. log_frequency is specified and that number of traces has been collected
  if (((count_ == 0) && (collected_ == sample_)) ||
      ((log_frequency_ != 0) && (sample_in_stream_ >= log_frequency_))) {
    // Reset variables and release lock before saving to file
    sample_in_stream_ = 0;
    std::stringstream stream;
    trace_stream_.swap(stream);
    lock.unlock();

    file_->SaveTraces(stream, true /* to_index_file */);
  }
}

TraceManager::TraceSetting::TraceSetting()
    : level_(TRITONSERVER_TRACE_LEVEL_DISABLED), rate_(0), count_(-1),
      log_frequency_(0), sample_(0), created_(0), collected_(0),
      sample_in_stream_(0)
{
  invalid_reason_ = "Setting hasn't been initialized";
}

TraceManager::TraceSetting::TraceSetting(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    const int32_t count, const uint32_t log_frequency,
    const std::shared_ptr<TraceFile>& file)
    : level_(level), rate_(rate), count_(count), log_frequency_(log_frequency),
      file_(file), sample_(0), created_(0), collected_(0), sample_in_stream_(0)
{
  if (level_ == TRITONSERVER_TRACE_LEVEL_DISABLED) {
    invalid_reason_ = "tracing is disabled";
  } else if (rate_ == 0) {
    invalid_reason_ = "sample rate must be non-zero";
  } else if (file_->FileName().empty()) {
    invalid_reason_ = "trace file name is not given";
  }
}

TraceManager::TraceSetting::~TraceSetting()
{
  // If log frequency is set, should log the remaining traces to indexed file.
  if (sample_in_stream_ != 0) {
    file_->SaveTraces(trace_stream_, (log_frequency_ != 0));
  }
}

}}}  // namespace triton::developer_tools::server
