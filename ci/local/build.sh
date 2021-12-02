#!/bin/bash

set -e

REPODIR=$(cd $(dirname $0)/../../; pwd)

EXAMPLE_TAG=rapids_triton_identity \
  TEST_TAG=rapids_triton_identity_test \
  $REPODIR/build.sh
if [ -z $CUDA_VISIBLE_DEVICES ]
then
  docker run --gpus all --rm rapids_triton_identity_test
else
  docker run --gpus $CUDA_VISIBLE_DEVICES --rm rapids_triton_identity_test
fi
EXAMPLE_TAG=rapids_triton_identity:cpu \
  TEST_TAG=rapids_triton_identity_test:cpu \
  $REPODIR/build.sh --cpu-only
docker run -v "${REPODIR}/qa/logs:/qa/logs" --gpus all --rm rapids_triton_identity_test:cpu
