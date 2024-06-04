#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <vector>

#include "triton/developer_tools/onnxruntime_c_api.h"

// Utility function to check and print error messages
void
check_status(OrtStatus* status)
{
  if (status != NULL) {
    const char* msg =
        OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(status);
    printf("Error: %s\n", msg);
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
    exit(1);
  }
}

struct InferContext {
  OrtSession* session_;
  size_t num_input_nodes_;
  OrtAllocator* allocator_;
  char* input_name_;

  OrtTypeInfo* type_info_;
  ONNXTensorElementDataType type_;
  const OrtTensorTypeAndShapeInfo* tensor_info_;

  size_t num_dims_;
  int64_t* input_node_dims_;

  float* input_tensor_values_;
  char* output_name_;

  OrtMemoryInfo* memory_info_;

  OrtValue* input_tensor_;
  OrtValue* output_tensor_;

  const char* input_names_[1];
  const char* output_names_[1];
};

int
main()
{
  // Initialize the ONNX Runtime API
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  // Create ONNX Runtime environment
  OrtEnv* env;
  check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // Create session options and enable CUDA as the execution provider
  OrtSessionOptions* session_options;
  check_status(g_ort->CreateSessionOptions(&session_options));
  //check_status(g_ort->SetIntraOpNumThreads(session_options, 1));
  //check_status(g_ort->SetInterOpNumThreads(session_options, 1));
  //check_status(g_ort->SetSessionGraphOptimizationLevel(session_options, GraphOptimizationLevel::ORT_ENABLE_EXTENDED));

  // Could this be the stream?
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  // cuda_options.has_user_compute_stream = stream != nullptr ? 1 : 0;
  cuda_options.has_user_compute_stream = 0;
  // cuda_options.user_compute_stream =
  //     stream != nullptr ? (void*)stream : nullptr,
  cuda_options.user_compute_stream = nullptr;
  cuda_options.default_memory_arena_cfg = nullptr;

  check_status(g_ort->SessionOptionsAppendExecutionProvider_CUDA(
      session_options, &cuda_options));  // 0 for the default GPU device

  std::vector<std::string> model_names{"bmode_perspective", "aortic_stenosis", "plax_chamber"};
  std::vector<InferContext> infer_contexts;

  for(const auto model_name: model_names) {
    std::string model_path = std::string("/work/server/build/install/bin/holoscan_models/" + model_name + "/1/model.onnx");
    std::cout << "Loading model from " << model_path << std::endl;
    InferContext context;
    check_status(
      g_ort->CreateSession(env, model_path.c_str(), session_options, &context.session_));

  check_status(g_ort->GetAllocatorWithDefaultOptions(&context.allocator_));
  check_status(g_ort->SessionGetInputCount(context.session_, &context.num_input_nodes_));
  check_status(g_ort->SessionGetInputName(context.session_, 0, context.allocator_, &context.input_name_));

  check_status(g_ort->SessionGetInputTypeInfo(context.session_, 0, &context.type_info_));
  
  check_status(g_ort->CastTypeInfoToTensorInfo(context.type_info_, &context.tensor_info_));
  check_status(g_ort->GetTensorElementType(context.tensor_info_, &context.type_));

  check_status(g_ort->GetDimensionsCount(context.tensor_info_, &context.num_dims_));
  context.input_node_dims_ = (int64_t*)malloc(context.num_dims_ * sizeof(int64_t));
  check_status(g_ort->GetDimensions(context.tensor_info_, context.input_node_dims_, context.num_dims_));

  // Prepare input data (example for a single float input)
  size_t elem_cnt = 1;
  for (size_t i = 0; i < context.num_dims_; i++) {
    if (context.input_node_dims_[i] < 0) {
      context.input_node_dims_[i] = 1;
    }
    elem_cnt *=  context.input_node_dims_[i];
  }
  context.input_tensor_values_ = (float*)malloc(elem_cnt * sizeof(float));
  for (size_t i = 0; i < elem_cnt; i++) {
    context.input_tensor_values_[i] = 1.0 * (std::rand()%256);
  }

  check_status(g_ort->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &context.memory_info_));

  context.input_tensor_ = NULL;
  check_status(g_ort->CreateTensorWithDataAsOrtValue(
      context.memory_info_, context.input_tensor_values_, elem_cnt * sizeof(float),
      context.input_node_dims_, context.num_dims_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &context.input_tensor_));

  int is_tensor;
  check_status(g_ort->IsTensor(context.input_tensor_, &is_tensor));

  // Prepare output data
  check_status(
      g_ort->SessionGetOutputName(context.session_, 0, context.allocator_, &context.output_name_));

  context.output_tensor_ = NULL;
  context.input_names_[0] = context.input_name_;
  context.output_names_[0] = context.output_name_;

  infer_contexts.push_back(context);
  }

  std::chrono::duration<double, std::milli> total_elapsed;

  // Warm-up
  for (int i = 0; i < 10000; i++) {
    check_status(g_ort->Run(
        infer_contexts[0].session_, NULL, infer_contexts[0].input_names_, (const OrtValue* const*)&infer_contexts[0].input_tensor_, 1,
        infer_contexts[0].output_names_, 1, &infer_contexts[0].output_tensor_));
    check_status(g_ort->Run(
        infer_contexts[1].session_, NULL, infer_contexts[1].input_names_, (const OrtValue* const*)&infer_contexts[1].input_tensor_, 1,
        infer_contexts[1].output_names_, 1, &infer_contexts[1].output_tensor_));
    check_status(g_ort->Run(
        infer_contexts[2].session_, NULL, infer_contexts[2].input_names_, (const OrtValue* const*)&infer_contexts[2].input_tensor_, 1,
        infer_contexts[2].output_names_, 1, &infer_contexts[2].output_tensor_));

  }


  for (int i = 0; i < 100; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    {
      nvtx3::scoped_range loop{"three inferences"};  // Range for iteration

      {
        nvtx3::scoped_range loop{"first inferences"};  // Range for iteration
        check_status(g_ort->Run(
        infer_contexts[0].session_, NULL, infer_contexts[0].input_names_, (const OrtValue* const*)&infer_contexts[0].input_tensor_, 1,
        infer_contexts[0].output_names_, 1, &infer_contexts[0].output_tensor_));
      }
      {
        nvtx3::scoped_range loop{"second inferences"};  // Range for iteration
        check_status(g_ort->Run(
        infer_contexts[1].session_, NULL, infer_contexts[1].input_names_, (const OrtValue* const*)&infer_contexts[1].input_tensor_, 1,
        infer_contexts[1].output_names_, 1, &infer_contexts[1].output_tensor_));
      }
      {
        nvtx3::scoped_range loop{"third inferences"};  // Range for iteration
        check_status(g_ort->Run(
        infer_contexts[2].session_, NULL, infer_contexts[2].input_names_, (const OrtValue* const*)&infer_contexts[2].input_tensor_, 1,
        infer_contexts[2].output_names_, 1, &infer_contexts[2].output_tensor_));
      }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    total_elapsed += end - start;
    std::cout << "Running Avg Elapsed time: " << total_elapsed.count() / i
              << " ms" << std::endl;
  }

  // Output the elapsed time
  std::cout << "Avg Elapsed time: " << total_elapsed.count() / 100 << " ms"
            << std::endl;


  // Retrieve the output data
  // float* output_data;
  // check_status(g_ort->GetTensorMutableData(output_tensor0,
  // (void**)&output_data));

  // Print the results
  // printf("Output values:\n");
  // for (size_t i = 0; i < 1000; i++) {
  //    printf("%f\n", output_data[i]);
  //}

  // Clean up

  for (auto context: infer_contexts) {
    g_ort->ReleaseValue(context.output_tensor_);

  g_ort->ReleaseValue(context.input_tensor_);
  g_ort->ReleaseMemoryInfo(context.memory_info_);
  free(context.input_tensor_values_);
  free(context.input_node_dims_);
  g_ort->ReleaseTypeInfo(context.type_info_);
  g_ort->ReleaseSession(context.session_);

  }
  
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);

  return 0;
}
