// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <cstdint>
#include <string>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/logging.hpp>
#include <triton/core/tritonbackend.h>

namespace triton { namespace backend { namespace rapids {
  inline auto
  get_backend_name(TRITONBACKEND_Backend& backend)
  {
    const char* cname;
    triton_check(TRITONBACKEND_BackendName(&backend, &cname));
    return std::string(cname);
  }

  namespace {
    struct backend_version {
      std::uint32_t major;
      std::uint32_t minor;
    };
  }

  inline auto
  check_backend_version(TRITONBACKEND_Backend& backend)
  {
    auto version = backend_version{};
    triton_check(TRITONBACKEND_ApiVersion(&version.major, &version.minor));

    log_info(__FILE__, __LINE__) << "Triton TRITONBACKEND API version: "
                                 << version.major
                                 << "."
                                 << version.minor;

    auto name = get_backend_name(backend);

    log_info(__FILE__, __LINE__) << "'"
                                 << name 
                                 << "' TRITONBACKEND API version: "
                                 << TRITONBACKEND_API_VERSION_MAJOR
                                 << "."
                                 << TRITONBACKEND_API_VERSION_MINOR;

    return (
        (version.major == TRITONBACKEND_API_VERSION_MAJOR) &&
        (version.minor >= TRITONBACKEND_API_VERSION_MINOR));
  }
}}}  // namespace triton::backend::rapids
