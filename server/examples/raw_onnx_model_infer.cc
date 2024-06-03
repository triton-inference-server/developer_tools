#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <nvtx3/nvtx3.hpp>

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
  check_status(g_ort->SetIntraOpNumThreads(session_options, 1));
  check_status(g_ort->SetInterOpNumThreads(session_options, 1));
  check_status(g_ort->SetSessionGraphOptimizationLevel(session_options, GraphOptimizationLevel::ORT_ENABLE_EXTENDED));

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

  // Load the ONNX model
  const char* model_path =
      "/work/server/build/install/bin/models/resnet50/1/resnet50-1.2.onnx";
  OrtSession* session;
  check_status(
      g_ort->CreateSession(env, model_path, session_options, &session));

  // Get input and output information
  size_t num_input_nodes;
  OrtAllocator* allocator;
  check_status(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  check_status(g_ort->SessionGetInputCount(session, &num_input_nodes));
  char* input_name;
  check_status(g_ort->SessionGetInputName(session, 0, allocator, &input_name));

  OrtTypeInfo* type_info;
  check_status(g_ort->SessionGetInputTypeInfo(session, 0, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_info;
  check_status(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));

  ONNXTensorElementDataType type;
  check_status(g_ort->GetTensorElementType(tensor_info, &type));

  size_t num_dims;
  check_status(g_ort->GetDimensionsCount(tensor_info, &num_dims));
  int64_t* input_node_dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
  check_status(g_ort->GetDimensions(tensor_info, input_node_dims, num_dims));

  // Prepare input data (example for a single float input)
  size_t elem_cnt = 1;
  for (size_t i = 0; i < num_dims; i++) {
    elem_cnt *= input_node_dims[i];
  }
  float* input_tensor_values = (float*)malloc(elem_cnt * sizeof(float));
  for (size_t i = 0; i < elem_cnt; i++) {
    input_tensor_values[i] = (float)i;
  }

  OrtMemoryInfo* memory_info;
  check_status(g_ort->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  OrtValue* input_tensor = NULL;
  check_status(g_ort->CreateTensorWithDataAsOrtValue(
      memory_info, input_tensor_values, elem_cnt * sizeof(float),
      input_node_dims, num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &input_tensor));

  int is_tensor;
  check_status(g_ort->IsTensor(input_tensor, &is_tensor));

  // Prepare output data
  char* output_name;
  check_status(
      g_ort->SessionGetOutputName(session, 0, allocator, &output_name));

  // Run the model
  OrtValue* output_tensor0 = NULL;
  OrtValue* output_tensor1 = NULL;
  OrtValue* output_tensor2 = NULL;

  const char* input_names[] = {input_name};
  const char* output_names[] = {output_name};
  std::chrono::duration<double, std::milli> total_elapsed;

  // Warm-up
  for (int i = 0; i < 10000; i++) {
    check_status(g_ort->Run(
        session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
        output_names, 1, &output_tensor0));
    check_status(g_ort->Run(
        session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
        output_names, 1, &output_tensor1));
    check_status(g_ort->Run(
        session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
        output_names, 1, &output_tensor2));
  }


  for (int i = 0; i < 100; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    {
      nvtx3::scoped_range loop{"three inferences"};  // Range for iteration

      {
        nvtx3::scoped_range loop{"first inferences"};  // Range for iteration
        check_status(g_ort->Run(
            session, NULL, input_names, (const OrtValue* const*)&input_tensor,
            1, output_names, 1, &output_tensor0));
      }
      {
        nvtx3::scoped_range loop{"second inferences"};  // Range for iteration
        check_status(g_ort->Run(
            session, NULL, input_names, (const OrtValue* const*)&input_tensor,
            1, output_names, 1, &output_tensor1));
      }
      {
        nvtx3::scoped_range loop{"third inferences"};  // Range for iteration
        check_status(g_ort->Run(
            session, NULL, input_names, (const OrtValue* const*)&input_tensor,
            1, output_names, 1, &output_tensor2));
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
  g_ort->ReleaseValue(output_tensor0);
  g_ort->ReleaseValue(output_tensor1);
  g_ort->ReleaseValue(output_tensor2);

  g_ort->ReleaseValue(input_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  free(input_tensor_values);
  free(input_node_dims);
  g_ort->ReleaseTypeInfo(type_info);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);

  return 0;
}
