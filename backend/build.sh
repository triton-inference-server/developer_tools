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

REPODIR=$(cd $(dirname $0); pwd)

NUMARGS=$#
ARGS=$*
VALIDTARGETS="example tests"
VALIDFLAGS="--cpu-only -g -h --help"
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   example          - build the identity backend example
   tests            - build container(s) with unit tests
 and <flag> is:
   -g               - build for debug
   -h               - print this text
   --cpu-only       - build CPU-only versions of targets
   --tag-commit     - tag docker images based on current git commit

 default action (no args) is to build all targets
 The following environment variables are also accepted to allow further customization:
   BASE_IMAGE       - Base image for Docker images
   TRITON_VERSION   - Triton version to use for build
   EXAMPLE_TAG      - The tag to use for the server image
   TEST_TAG         - The tag to use for the test image
"

BUILD_TYPE=Release
TRITON_ENABLE_GPU=ON
DOCKER_ARGS=""

export DOCKER_BUILDKIT=1

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function completeBuild {
    (( ${NUMARGS} == 0 )) && return
    for a in ${ARGS}; do
        if (echo " ${VALIDTARGETS} " | grep -q " ${a} "); then
          false; return
        fi
    done
    true
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Long arguments
LONG_ARGUMENT_LIST=(
    "cpu-only"
    "tag-commit"
)

# Short arguments
ARGUMENT_LIST=(
    "g"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s," "${LONG_ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "$(printf "%s" "${ARGUMENT_LIST[@]}")" \
    -- "$@"
)

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$opts"

while true
do
  case "$1" in
    -g | --debug )
      BUILD_TYPE=Debug
      ;;
    --cpu-only )
      TRITON_ENABLE_GPU=OFF
      ;;
    --tag-commit )
      [ -z $EXAMPLE_TAG ] \
        && EXAMPLE_TAG="rapids_triton_identity:$(cd $REPODIR; git rev-parse --short HEAD)" \
        || true
      [ -z $TEST_TAG ] \
        && TEST_TAG="rapids_triton_identity_test:$(cd $REPODIR; git rev-parse --short HEAD)" \
        || true
      ;;
    --)
      shift
      break
      ;;
  esac
  shift
done

if [ -z $EXAMPLE_TAG ]
then
  EXAMPLE_TAG='rapids_triton_identity'
fi
if [ -z $TEST_TAG ]
then
  TEST_TAG='rapids_triton_identity_test'
fi

DOCKER_ARGS="$DOCKER_ARGS --build-arg BUILD_TYPE=${BUILD_TYPE}"
DOCKER_ARGS="$DOCKER_ARGS --build-arg TRITON_ENABLE_GPU=${TRITON_ENABLE_GPU}"

if [ ! -z $BASE_IMAGE ]
then
  DOCKER_ARGS="$DOCKER_ARGS --build-arg BASE_IMAGE=${BASE_IMAGE}"
fi
if [ ! -z $TRITON_VERSION ]
then
  DOCKER_ARGS="$DOCKER_ARGS --build-arg TRITON_VERSION=${TRITON_VERSION}"
fi

if completeBuild || hasArg example
then
  BACKEND=1
  DOCKER_ARGS="$DOCKER_ARGS --build-arg BUILD_EXAMPLE=ON"
fi

if completeBuild || hasArg tests
then
  TESTS=1
  DOCKER_ARGS="$DOCKER_ARGS --build-arg BUILD_TESTS=ON"
fi

if [ $BACKEND -eq 1 ]
then
  docker build \
    $DOCKER_ARGS \
    -t "$EXAMPLE_TAG" \
    $REPODIR
fi

if [ $TESTS -eq 1 ]
then
  docker build \
    $DOCKER_ARGS \
    -t "$EXAMPLE_TAG" \
    --target test-stage \
    -t "$TEST_TAG" \
    $REPODIR
fi
