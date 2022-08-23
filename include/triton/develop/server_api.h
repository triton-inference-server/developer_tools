// [DO NOT MERGE] mock for testing setup

#pragma once

#include <vector>
#include <set>
#include <string>
#include <exception>
#include "triton/core/tritonserver.h"

namespace triton { namespace high_level { namespace server_api {

class Error {

};

struct TritonException : std::exception {
 public:
  enum class Code {
    Unknown,
    Internal,
    NotFound,
    InvalidArg,
    Unavailable,
    Unsupported,
    AlreadyExists
  };

  TritonException() : code_(Code::Unknown), msg_("encountered unknown error") {}

  TritonException(const Code code, const std::string& msg)
    : code_(code), msg_(msg)
  {
  }
  virtual char const* what() const noexcept override { return msg_.c_str(); }

 private:
  const Code code_;
  const std::string msg_;
};

//==============================================================================
/// Server parameters.
///
struct ServerParams {
  explicit ServerParams(std::vector<std::string> model_repository_paths)
      : model_repository_paths(model_repository_paths)
  {
    server_id = "triton";
    backend_dir = "/opt/tritonserver/backends";
    repo_agent_dir = "/opt/tritonserver/repoagents";
    disable_auto_complete_config = false;
  }

  std::vector<std::string> model_repository_paths;
  std::string server_id;
  std::string backend_dir;
  std::string repo_agent_dir;
  bool disable_auto_complete_config;
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
 private:
  std::set<std::string> loaded_models_;
};

}}}  // namespace triton::high_level::server_api
