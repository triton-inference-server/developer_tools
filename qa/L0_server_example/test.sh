# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

bash -x ../install_test_dependencies_and_build.sh
export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG=`pwd`/client.log
SIMPLE_ADDSUB_ASYNC_INFER_CLIENT=/opt/tritonserver/developer_tools/server/build/install/bin/simple_addsub_async_infer
ADDSUB_STRING_ASYNC_INFER_CLIENT=/opt/tritonserver/developer_tools/server/build/install/bin/addsub_string_async_infer
SQUARE_ASYNC_INFER_CLIENT=/opt/tritonserver/developer_tools/server/build/install/bin/square_async_infer

RET=0

# Prepare required models for the examples
mkdir models
cp -r ../L0_server_unit_test/models/add_sub* ./models/.
git clone --single-branch --depth=1 -b ${TRITON_SERVER_BRANCH_NAME} https://github.com/triton-inference-server/server.git
cp -r server/docs/examples/model_repository/simple ./models/.
# Copy over the decoupled model placed in the python_backend repository.
git clone --single-branch --depth=1 -b ${PYTHON_BACKEND_REPO_TAG} https://github.com/triton-inference-server/python_backend.git
mkdir -p ./models/square_int32/1
cp python_backend/examples/decoupled/square_model.py ./models/square_int32/1/model.py
cp python_backend/examples/decoupled/square_config.pbtxt ./models/square_int32/config.pbtxt

# Must explicitly set LD_LIBRARY_PATH so that the test can find
# libtritonserver.so.
LD_LIBRARY_PATH=/opt/tritonserver/lib:${LD_LIBRARY_PATH}

set +e

for i in \
    $SIMPLE_ADDSUB_ASYNC_INFER_CLIENT \
    $ADDSUB_STRING_ASYNC_INFER_CLIENT \
    $SQUARE_ASYNC_INFER_CLIENT \
    ; do
    BASE=$(basename -- $i)
    SUFFIX="${BASE%.*}"

    if [ $i == $SIMPLE_ADDSUB_ASYNC_INFER_CLIENT ]; then
        # Enforce I/O to be in specific memory type
        for MEM_TYPE in system pinned gpu ; do
            $i -v -m $MEM_TYPE >> $CLIENT_LOG.${SUFFIX}.$MEM_TYPE 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG.${SUFFIX}.$MEM_TYPE
                RET=1
            fi
        done
    else
        $i -v >> ${CLIENT_LOG}.${SUFFIX} 2>&1
        if [ $? -ne 0 ]; then
            cat ${CLIENT_LOG}.${SUFFIX}
            RET=1
        fi
    fi
done

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
