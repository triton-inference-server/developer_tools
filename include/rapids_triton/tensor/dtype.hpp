/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <triton/core/tritonserver.h>

using DType = TRITONSERVER_DataType;
using DTypeBool = TRITONSERVER_TYPE_BOOL;
using DTypeUint8 = TRITONSERVER_TYPE_UINT8;
using DTypeUint16 = TRITONSERVER_TYPE_UINT16;
using DTypeUint32 = TRITONSERVER_TYPE_UINT32;
using DTypeUint64 = TRITONSERVER_TYPE_UINT64;
using DTypeInt8 = TRITONSERVER_TYPE_INT8;
using DTypeInt16 = TRITONSERVER_TYPE_INT16;
using DTypeInt32 = TRITONSERVER_TYPE_INT32;
using DTypeInt64 = TRITONSERVER_TYPE_INT64;
using DTypeFloat32 = TRITONSERVER_TYPE_FP32;
using DTypeFloat64 = TRITONSERVER_TYPE_FP64;


template <DType D>
struct TritonType {
};

template <typename T>
struct TritonDtype {
};

template <>
struct TritonType<DTypeBool> {
  typedef bool type;
};

template <>
struct TritonDtype<bool> {
  static constexpr DType value = DTypeBool;
};

template <>
struct TritonType<DTypeUint8> {
  typedef uint8_t type;
};

template <>
struct TritonDtype<uint8_t> {
  static constexpr DType value = DTypeUint8;
};

template <>
struct TritonType<DTypeUint16> {
  typedef uint16_t type;
};

template <>
struct TritonDtype<uint16_t> {
  static constexpr DType value = DTypeUint16;
};

template <>
struct TritonType<DTypeUint32> {
  typedef uint32_t type;
};

template <>
struct TritonDtype<uint32_t> {
  static constexpr DType value = DTypeUint32;
};

template <>
struct TritonType<DTypeUint64> {
  typedef uint64_t type;
};

template <>
struct TritonDtype<uint64_t> {
  static constexpr DType value = DTypeUint64;
};

template <>
struct TritonType<DTypeInt8> {
  typedef int8_t type;
};

template <>
struct TritonDtype<int8_t> {
  static constexpr DType value = DTypeInt8;
};

template <>
struct TritonType<DTypeInt16> {
  typedef int16_t type;
};

template <>
struct TritonDtype<int16_t> {
  static constexpr DType value = DTypeInt16;
};

template <>
struct TritonType<DTypeInt32> {
  typedef int32_t type;
};

template <>
struct TritonDtype<int32_t> {
  static constexpr DType value = DTypeInt32;
};

template <>
struct TritonType<DTypeInt64> {
  typedef int64_t type;
};

template <>
struct TritonDtype<int64_t> {
  static constexpr DType value = DTypeInt64;
};

template <>
struct TritonType<DTypeFloat32> {
  typedef float type;
};

template <>
struct TritonDtype<float> {
  static constexpr DType value = DTypeFloat32;
};

template <>
struct TritonType<TRITONSERVER_TYPE_FP64> {
  typedef double type;
};

template <>
struct TritonDtype<double> {
  static constexpr DType value = DTypeFloat64;
};

// TODO(whicks): Correctly handle const versions of types
