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
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#else
#include <triton_backend/cpu_only/cuda_runtime_replacement.hpp>
#endif

#include <triton/backend/backend_output_responder.h>
#include <triton_backend/build_control.hpp>
#include <triton_backend/exceptions.hpp>
#include <triton_backend/memory/buffer.hpp>
#include <triton_backend/tensor/dtype.hpp>
#include <triton_backend/triton/device.hpp>
#include <triton_backend/utils/narrow.hpp>

namespace triton {
namespace backend {
namespace dev_tools {
template <typename T>
struct BaseTensor {
  using size_type = typename Buffer<T>::size_type;

  BaseTensor() : shape_{}, buffer_{} {}
  BaseTensor(std::vector<size_type> const& shape, Buffer<T>&& buffer)
    : shape_(shape), buffer_{std::move(buffer)}
  {
  }

  virtual ~BaseTensor() = 0;

  /**
   * @brief Construct a BaseTensor from a collection of buffers
   *
   * Given a collection of buffers, collate them all into one buffer stored in
   * a new BaseTensor
   */
  template <typename Iter>
  BaseTensor(std::vector<size_type> const& shape,
             Iter begin,
             Iter end,
             MemoryType mem_type,
             device_id_t device,
             cudaStream_t stream)
    : shape_(shape), buffer_([&begin, &end, &mem_type, &device, &stream]() {
        auto total_size = std::transform_reduce(
          begin, end, size_type{}, std::plus<>{}, [](auto&& buffer) { return buffer.size(); });

        auto result = Buffer<T>(total_size, mem_type, device, stream);

        std::accumulate(begin, end, size_type{}, [&result](auto offset, auto& buffer) {
          copy(result, buffer, offset);
          return offset + buffer.size();
        });
        return result;
      }())
  {
  }

  auto const& shape() const { return shape_; }
  auto size() const { return buffer_.size(); }
  auto data() const { return buffer_.data(); }
  auto& buffer() { return buffer_; }

  auto constexpr dtype() { return TritonDtype<T>::value; }
  auto mem_type() const { return buffer_.mem_type(); }
  auto stream() const { return buffer_.stream(); }
  auto device() const { return buffer_.device(); }

  void stream_synchronize() const
  {
    if (mem_type() == DeviceMemory) { buffer_.stream_synchronize(); }
  }

  void set_stream(cudaStream_t new_stream) { buffer_.set_stream(new_stream); }

 private:
  std::vector<size_type> shape_;
  Buffer<T> buffer_;
};

template <typename T>
BaseTensor<T>::~BaseTensor()
{
}

template <typename T>
struct Tensor final : BaseTensor<T> {
  Tensor() : BaseTensor<T>{} {}
  Tensor(std::vector<typename BaseTensor<T>::size_type> const& shape, Buffer<T>&& buffer)
    : BaseTensor<T>(shape, std::move(buffer))
  {
  }

  template <typename Iter>
  Tensor(std::vector<typename BaseTensor<T>::size_type> const& shape,
         Iter begin,
         Iter end,
         MemoryType mem_type,
         device_id_t device,
         cudaStream_t stream)
    : BaseTensor<T>(shape, begin, end, mem_type, device, stream)
  {
  }
};

template <typename T>
struct OutputTensor final : BaseTensor<T> {
  OutputTensor(std::vector<typename BaseTensor<T>::size_type>&& shape,
               Buffer<T>&& buffer,
               std::string const& name,
               std::shared_ptr<BackendOutputResponder> responder)
    : BaseTensor<T>(std::move(shape), std::move(buffer)), name_{name}, responder_{responder}
  {
  }
  /**
   * @brief Prepare final output data from this tensor for responding to
   * request
   *
   * This method *must* be called by triton_backend backends on all of their
   * output tensors before returning from their `predict` methods. Because we
   * cannot know a priori what names backends might have for their tensors
   * and what types will be stored in those tensors, the triton_backend
   * library cannot store references to those tensors that might otherwise be
   * used to finalize them.
   */
  void finalize()
  {
    auto& shape       = BaseTensor<T>::shape();
    auto triton_shape = std::vector<std::int64_t>{};
    triton_shape.reserve(shape.size());
    std::transform(
      std::begin(shape), std::end(shape), std::back_inserter(triton_shape), [](auto& val) {
        return narrow<int64_t>(val);
      });

    // Must call the following because BackendOutputResponder does not expose
    // its stream, so we cannot be certain that our data is not being
    // processed on another stream.
    BaseTensor<T>::stream_synchronize();
    responder_->ProcessTensor(name_.c_str(),
                              TritonDtype<T>::value,
                              triton_shape,
                              reinterpret_cast<char*>(BaseTensor<T>::data()),
                              BaseTensor<T>::mem_type(),
                              BaseTensor<T>::device());
  }

 private:
  std::string name_;
  std::shared_ptr<BackendOutputResponder> responder_;
};

template <typename T,
          typename U,
          typename = std::enable_if_t<std::is_same_v<std::remove_const_t<U>, T>>>
void copy(BaseTensor<T>& dst, BaseTensor<U>& src)
{
  copy(dst.buffer(), src.buffer());
}

/**
 * @brief Copy data from src Tensor into buffers indicated by iterators
 *
 * This method is provided to assist with distributing data from a single
 * Tensor into many smaller buffers which have been set up to receive a part
 * of the data from the src Tensor
 */
template <typename T, typename Iter>
void copy(Iter begin, Iter end, BaseTensor<T>& src)
{
  std::accumulate(begin, end, typename BaseTensor<T>::size_type{}, [&src](auto offset, auto& dst) {
    auto end_offset = offset + dst.size();
    copy(dst.buffer(), src.buffer(), offset, end_offset);
    return end_offset;
  });
}

}  // namespace dev_tools
}  // namespace backend
}  // namespace triton
