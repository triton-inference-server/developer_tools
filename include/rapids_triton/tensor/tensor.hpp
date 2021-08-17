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

#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/tensor/dtype.hpp>

namespace triton { namespace backend { namespace rapids {
  template <typename T>
  struct Tensor {
   using size_type = typename Buffer<T>::size_type;

   Tensor() : shape_{}, buffer_{} {}

   /**
    * @brief Construct a Tensor from a collection of buffers
    *
    * Given a collection of buffers, collate them all into one buffer stored in
    * a new Tensor
    */
   template <typename Iter>
   Tensor(std::vector<size_type> const& shape, Iter begin, Iter end, MemoryType mem_type, cudaStream_t stream=0) : 
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

   auto constexpr dtype() const { return TritonDtype<T>::value; }
   auto mem_type() const { return data.mem_type(); }
   auto stream() const { return data.stream(); }

   private:
     std::vector<size_type> shape_;
     Buffer<T> buffer_;
  };

  template<typename T>
  void copy(Tensor<std::remove_const_t<T>> dst, Tensor<T> src) {
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
  copy(Iter begin, Iter end, Tensor<T> src) {
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

  /**
   * @brief A simple struct containing only a string used to name a Tensor and
   * the Tensor itself
   */
  template <typename T>
  struct NamedTensor {
    std::string name;
    Tensor<T> tensor;
  };
}}}  // namespace triton::backend::rapids
