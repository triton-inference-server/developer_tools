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
#include <cstdint>
#include <iostream>
#include <rapids_triton/utils/const_agnostic.hpp>

namespace triton {
namespace backend {
namespace rapids {

using DType                 = TRITONSERVER_DataType;
auto constexpr DTypeBool    = TRITONSERVER_TYPE_BOOL;
auto constexpr DTypeUint8   = TRITONSERVER_TYPE_UINT8;
auto constexpr DTypeChar    = DTypeUint8;
auto constexpr DTypeByte    = DTypeUint8;
auto constexpr DTypeUint16  = TRITONSERVER_TYPE_UINT16;
auto constexpr DTypeUint32  = TRITONSERVER_TYPE_UINT32;
auto constexpr DTypeUint64  = TRITONSERVER_TYPE_UINT64;
auto constexpr DTypeInt8    = TRITONSERVER_TYPE_INT8;
auto constexpr DTypeInt16   = TRITONSERVER_TYPE_INT16;
auto constexpr DTypeInt32   = TRITONSERVER_TYPE_INT32;
auto constexpr DTypeInt64   = TRITONSERVER_TYPE_INT64;
auto constexpr DTypeFloat32 = TRITONSERVER_TYPE_FP32;
auto constexpr DTypeFloat64 = TRITONSERVER_TYPE_FP64;

template <DType D>
struct TritonType {
};

template <typename T, typename = void>
struct TritonDtype {
};

template <>
struct TritonType<DTypeBool> {
  typedef bool type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, bool>> {
  static constexpr DType value = DTypeBool;
};

template <>
struct TritonType<DTypeUint8> {
  typedef std::uint8_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::uint8_t>> {
  static constexpr DType value = DTypeUint8;
};

template <>
struct TritonType<DTypeUint16> {
  typedef std::uint16_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::uint16_t>> {
  static constexpr DType value = DTypeUint16;
};

template <>
struct TritonType<DTypeUint32> {
  typedef std::uint32_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::uint32_t>> {
  static constexpr DType value = DTypeUint32;
};

template <>
struct TritonType<DTypeUint64> {
  typedef std::uint64_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::uint64_t>> {
  static constexpr DType value = DTypeUint64;
};

template <>
struct TritonType<DTypeInt8> {
  typedef std::int8_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::int8_t>> {
  static constexpr DType value = DTypeInt8;
};

template <>
struct TritonType<DTypeInt16> {
  typedef std::int16_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::int16_t>> {
  static constexpr DType value = DTypeInt16;
};

template <>
struct TritonType<DTypeInt32> {
  typedef std::int32_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::int32_t>> {
  static constexpr DType value = DTypeInt32;
};

template <>
struct TritonType<DTypeInt64> {
  typedef std::int64_t type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, std::int64_t>> {
  static constexpr DType value = DTypeInt64;
};

template <>
struct TritonType<DTypeFloat32> {
  typedef float type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, float>> {
  static constexpr DType value = DTypeFloat32;
};

template <>
struct TritonType<TRITONSERVER_TYPE_FP64> {
  typedef double type;
};

template <typename T>
struct TritonDtype<T, const_agnostic_same_t<T, double>> {
  static constexpr DType value = DTypeFloat64;
};

inline std::ostream& operator<<(std::ostream& out, DType const& dtype)
{
  out << TRITONSERVER_DataTypeString(dtype);
  return out;
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
