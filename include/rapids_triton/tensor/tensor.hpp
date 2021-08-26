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
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

#include <rapids_triton/build_control.hpp>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/tensor/dtype.hpp>

namespace triton { namespace backend { namespace rapids {
  template <typename T>
  struct BaseTensor {
   using size_type = typename Buffer<T>::size_type;

   BaseTensor() : shape_{}, buffer_{} {}
   BaseTensor(std::vector<size_type>&& shape, Buffer&& buffer) : shape_(std::move(shape)), buffer_{std::move(buffer)} {}

   virtual ~BaseTensor() = 0;

   /**
    * @brief Construct a BaseTensor from a collection of buffers
    *
    * Given a collection of buffers, collate them all into one buffer stored in
    * a new BaseTensor
    */
   template <typename Iter>
   BaseTensor(std::vector<size_type> const& shape, Iter begin, Iter end, MemoryType mem_type, cudaStream_t stream=0) : 
     shape_(shape),
     buffer_([&begin, &end, &mem_type, &stream] () {
       auto total_size = std::transform_reduce(
          begin, end, std::plus<>{}, [](auto&& buffer) { return buffer.size(); }
       );

       auto result = Buffer<T>(total_size, mem_type, stream);

       std::reduce(begin, end, size_type{0}, [&result](auto&& buffer, auto offset) {
         copy(result, buffer, offset);
         return offset + buffer.size();
       });
       return result;
     }()) {}

   auto const& shape() const { return shape_; }
   auto data() { return buffer_.data(); }
   auto& buffer() { return buffer_; }

   auto dtype() constexpr { return TritonDtype<T>::value; }
   auto mem_type() const { return data.mem_type(); }
   auto stream() const { return data.stream(); }
   auto device() const { return data.device(); }

   void stream_synchronize() const {
     if constexpr (IS_GPU_BUILD) {
       if (mem_type() == DeviceMemory) {
         cuda_check(cudaStreamSynchronize(stream());
       }
     }
   }

   private:
     std::vector<size_type> shape_;
     Buffer<T> buffer_;
  };

  template<typename T>
  void copy(BaseTensor<std::remove_const_t<T>> dst, BaseTensor<T> src) {
    copy(dst.buffer(), src.buffer());
  }

  /**
   * @brief Copy data from src Tensor into buffers indicated by iterators
   *
   * This method is provided to assist with distributing data from a single
   * Tensor into many smaller buffers which have been set up to receive a part
   * of the data from the src Tensor
   */
  template<typename T, typename Iter>
  copy(Iter begin, Iter end, BaseTensor<T> src) {
    std::reduce(
      begin,
      end,
      decltype(*begin)::size_type{0},
      [&src] (auto&& buffer, auto offset) {
        auto end_offset = offset + buffer.size();
        copy(buffer, src.buffer(), offset, end_offset);
        return end_offset;
      }
  }

  template<typename T>
  struct Tensor final : BaseTensor<T> {
  };

  template<typename T>
  struct OutputTensor final : BaseTensor<T> {
    OutputTensor(std::vector<size_type>&& shape, Buffer&& buffer,
        std::string const& name, std::shared_pointer<BackendOutputResponder> responder) :
    BaseTensor<T>(std::move(shape), std::move(buffer)), name_{name}, responder_{responder}
    {}
    /**
     * @brief Prepare final output data from this tensor for responding to
     * request
     *
     * This method *must* be called by rapids_triton backends on all of their
     * output tensors before returning from their `predict` methods. Because we
     * cannot known a priori what names backends might have for their tensors
     * and what types will be stored in those tensors, the rapids_triton
     * library cannot store references to those tensors that might otherwise be
     * used to finalize them.
     */
    void finalize() {
      // Must call the following because BackendOutputResponder does not expose
      // its stream, so we cannot be certain that our data is not being
      // processed on another stream.
      stream_synchronize();
      responder_->ProcessTensor(
        name_.c_str(),
        TritonDtype<T>::value,
        shape(),
        data(),
        mem_type(),
        device()
      );
    }

    private:
      std::shared_pointer<BackendOutputResponder> responder_;
      std::string name_;
  };
}}}  // namespace triton::backend::rapids
