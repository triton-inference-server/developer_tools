// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gtest/gtest.h"

#include "server_wrapper.h"
#include "triton/core/tritonserver.h"
#include <exception>

namespace tsw = triton::server::wrapper;

namespace {

class ServerWrapperTest : public ::testing::Test {
};

TEST_F(ServerWrapperTest, SanityCheck)
{
  // Sanity check that proper 'libtritonserver.so' is used
  uint32_t major = 0, minor = 0;
  auto err = TRITONSERVER_ApiVersion(&major, &minor);
  ASSERT_TRUE(err == nullptr) << "Unexpected error from API version call";
  ASSERT_EQ(major, TRITONSERVER_API_VERSION_MAJOR) << "Mismatch major version";
  ASSERT_GE(minor, TRITONSERVER_API_VERSION_MINOR) << "Older minor version";
}

TEST_F(ServerWrapperTest, StartServer)
{
  // Run server with invalid model repository
  try {
    tsw::TritonServer(tsw::ServerOptions({"/invalid_model_repository"}));
  } catch (std::exception& ex) {
    // check
    // [FIXME] should have Triton specific error reporting, either error object
    // or exception
    ASSERT_STREQ(ex.what(), "some error message");
  } catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
