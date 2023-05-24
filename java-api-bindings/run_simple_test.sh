#!/bin/bash


cd /tmp/build/javacpp-presets/tritonserverwrapper/platform/target
JAR_INSTALL_PATH="/workspace/install/java-api-bindings"
mkdir -p ${JAR_INSTALL_PATH}
cp tritonserverwrapper-platform-2.26-1.5.9-SNAPSHOT-shaded.jar ${JAR_INSTALL_PATH}/tritonserverwrapper-java-bindings.jar

cd /opt/tritonserver/javacpp-presets/tritonserverwrapper/samples/simple

cp /workspace/install/java-api-bindings/tritonserverwrapper-java-bindings.jar /opt/tritonserver/javacpp-presets/tritonserverwrapper/samples/simple/tritonserverwrapper-java-bindings.jar

cd /opt/tritonserver
git clone https://github.com/triton-inference-server/server.git
cd /opt/tritonserver/server/docs/examples
./fetch_models.sh

# cd /opt/tritonserver/javacpp-presets/tritonserverwrapper/samples/simple
# java -cp tritonserverwrapper-java-bindings.jar Simple.java -r /opt/tritonserver/server/docs/examples/model_repository

mkdir -p /workspace/triton/tmp/models
cp -r /opt/tritonserver/server/docs/examples/model_repository/*  /workspace/triton/tmp/models

