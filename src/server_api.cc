// [DO NOT MERGE] mock for testing setup

#include "server_api.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace high_level { namespace server_api {

namespace {

TritonException::Code ErrorCodeToExceptionCode(const TRITONSERVER_Error_Code code)
{
  switch (code)
  {
case TRITONSERVER_ERROR_UNKNOWN: return TritonException::Code::Unknown;
case TRITONSERVER_ERROR_INTERNAL: return TritonException::Code::Internal;
case TRITONSERVER_ERROR_NOT_FOUND: return TritonException::Code::NotFound;
case TRITONSERVER_ERROR_INVALID_ARG: return TritonException::Code::InvalidArg;
case TRITONSERVER_ERROR_UNAVAILABLE: return TritonException::Code::Unavailable;
case TRITONSERVER_ERROR_UNSUPPORTED: return TritonException::Code::Unsupported;
case TRITONSERVER_ERROR_ALREADY_EXISTS: return TritonException::Code::AlreadyExists;
  }
  return TritonException::Code::Unknown;
}
  
} // namespace


TritonServer::TritonServer(ServerParams server_params) {
  // Dummy implementation to make sure the library is properly linked
  auto err = TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_INVALID_ARG, "test error");
  if (err != nullptr) {
    auto ex = TritonException(ErrorCodeToExceptionCode(TRITONSERVER_ErrorCode(err)), TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    throw ex;
  }
}

TritonServer::~TritonServer() {}

Error TritonServer::LoadModel(const std::string model_name) { return Error(); }

Error TritonServer::UnloadModel(const std::string model_name) { return Error(); }

Error TritonServer::ModelConfig(
      std::string* model_config, const std::string model_name,
      const std::string& model_version) { return Error(); }

Error TritonServer::LoadedModels(std::set<std::string>* loaded_models) { return Error(); }

Error TritonServer::ModelIndex(std::string* repository_index) { return Error(); }

}}}  // namespace triton::high_level::server_api
