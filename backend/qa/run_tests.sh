#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

QA_DIR=$(cd $(dirname $0); pwd)
SERVER_ARGS=""
UUID="$(cat /proc/sys/kernel/random/uuid)"
CONTAINER_NAME="rapids_triton-ci-$UUID"
DOCKER_RUN=0
DOCKER_ARGS="-d -p 8000:8000 -p 8001:8001 -p 8002:8002 --name ${CONTAINER_NAME}"
TRITON_PID=""
LOG_DIR="${QA_DIR}/logs"
SERVER_LOG="${LOG_DIR}/${UUID}-server.log"

if [ ! -d "${LOG_DIR}" ]
then
  mkdir -p "${LOG_DIR}"
fi

if [ -z $MODEL_REPO ]
then
  MODEL_REPO="${QA_DIR}/L0_e2e/model_repository"
fi
MODEL_REPO="$(readlink -f $MODEL_REPO)"

DOCKER_ARGS="${DOCKER_ARGS} -v ${MODEL_REPO}:/models"

if [ -z $CPU_ONLY ] || [ $CPU_ONLY -eq 0 ]
then
  if [[ -v CUDA_VISIBLE_DEVICES ]]
  then
    if [ -z $CUDA_VISIBLE_DEVICES ]
    then
      CPU_ONLY=1
    else
      DOCKER_ARGS="${DOCKER_ARGS} --gpus ${CUDA_VISIBLE_DEVICES}"
    fi
  else
    DOCKER_ARGS="${DOCKER_ARGS} --gpus all"
  fi
else
  export CUDA_VISIBLE_DEVICES=""
fi

# If a Triton Docker image has been provided or no tritonserver executable is
# available, run the server via Docker
if [ ! -z $TRITON_IMAGE ] || ! command -v tritonserver
then
  DOCKER_RUN=1
  TRITON_IMAGE=${TRITON_IMAGE:-rapids_triton_identity}
  SERVER_ARGS="${SERVER_ARGS} --model-repository=/models"
else
  SERVER_ARGS="${SERVER_ARGS} --model-repository=${MODEL_REPO}"
fi

start_server() {
  if [ $DOCKER_RUN -eq 1 ]
  then
    docker run $DOCKER_ARGS $TRITON_IMAGE > /dev/null
  else
    tritonserver $SERVER_ARGS > $SERVER_LOG 2>&1 &
    TRITON_PID="$!"
  fi
}

start_server

if [ -z $TEST_EXE ]
then
  echo 'No TEST_EXE variable defined; skipping unit tests'
else
  if [ $DOCKER_RUN -eq 1 ]
  then
    docker exec $CONTAINER_NAME "$TEST_EXE"
  else
    "$TEST_EXE"
  fi
fi

finally() {
  if [ -z $TRITON_PID ]
  then
    docker logs $CONTAINER_NAME > $SERVER_LOG 2>&1
    docker rm -f $CONTAINER_NAME > /dev/null 2>&1
  else
    kill -15 $TRITON_PID
  fi
}

trap finally EXIT

pytest "$QA_DIR"
