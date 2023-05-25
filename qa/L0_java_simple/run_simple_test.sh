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

cd /tmp/build/javacpp-presets/tritondevelopertoolsserver/platform/target
JAR_INSTALL_PATH="/workspace/install/java-api-bindings"
mkdir -p ${JAR_INSTALL_PATH}
cp tritondevelopertoolsserver-platform-2.26-1.5.9-SNAPSHOT-shaded.jar ${JAR_INSTALL_PATH}/tritondevelopertoolsserver-java-bindings.jar

cd /opt/tritonserver/javacpp-presets/tritondevelopertoolsserver/samples/simple

cp /workspace/install/java-api-bindings/tritondevelopertoolsserver-java-bindings.jar /opt/tritonserver/javacpp-presets/tritondevelopertoolsserver/samples/simple/tritondevelopertoolsserver-java-bindings.jar

cd /opt/tritonserver
git clone https://github.com/triton-inference-server/server.git
cd /opt/tritonserver/server/docs/examples
./fetch_models.sh

# cd /opt/tritonserver/javacpp-presets/tritondevelopertoolsserver/samples/simple
# java -cp tritondevelopertoolsserver-java-bindings.jar Simple.java -r /opt/tritonserver/server/docs/examples/model_repository

mkdir -p /workspace/triton/tmp/models
cp -r /opt/tritonserver/server/docs/examples/model_repository/*  /workspace/triton/tmp/models

