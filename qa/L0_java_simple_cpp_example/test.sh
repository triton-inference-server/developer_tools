#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# set variables
CLIENT_LOG="client.log"
MODEL_REPO=$PWD/models
SAMPLES_REPO=$PWD/javacpp-presets/tritonserver/samples/simplecpp
TRITON_SERVER_REPO_TAG=${TRITON_SERVER_REPO_TAG:="main"}
TRITON_CLIENT_REPO_TAG=${TRITON_CLIENT_REPO_TAG:="main"}
TEST_HOME=$PWD

# generate models
rm -rf ${MODEL_REPO}
git clone --single-branch --depth=1 -b ${TRITON_SERVER_REPO_TAG} https://github.com/triton-inference-server/server.git
mkdir -p ${MODEL_REPO}
cp -r server/docs/examples/model_repository/simple ${MODEL_REPO}/simple

# use build script to generate .jar
git clone --single-branch --depth=1 -b ${TRITON_CLIENT_REPO_TAG} https://github.com/triton-inference-server/client.git
source client/src/java-api-bindings/scripts/install_dependencies_and_build.sh --enable-developer-tools-server

cd ${TEST_HOME}
# build javacpp-presets/tritonserver
set +e
rm -r javacpp-presets
git clone --single-branch --depth=1 -b ${JAVACPP_BRANCH_TAG} ${JAVACPP_BRANCH}
cd javacpp-presets
${MAVEN_PATH} clean install --projects .,tritonserver
${MAVEN_PATH} clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform.host
cd ..
set -e

rm -f *.log
RET=0

set +e
# Build SimpleCPP example
BASE_COMMAND="${MAVEN_PATH} clean compile -f ${SAMPLES_REPO} exec:java -Djavacpp.platform=linux-x86_64"
${BASE_COMMAND} -Dexec.args="-r ${MODEL_REPO}" >>${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    echo -e "Failed to run: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO}\""
    RET=1
fi

# Run SimpleCPP with generated jar
java -cp ${JAR_INSTALL_PATH}/tritonserver-java-bindings.jar ${SAMPLES_REPO}/SimpleCPP.java
if [ $? -ne 0 ]; then
    echo -e "Failed to run: java -cp ${JAR_INSTALL_PATH}/tritonserver-java-bindings.jar ${SAMPLES_REPO}/SimpleCPP.java -r ${MODEL_REPO}"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
