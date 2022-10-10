#!/bin/bash

set -e

REPODIR=$(cd $(dirname $0)/../../; pwd)

EXAMPLE_TAG=triton_dt_identity \
  TEST_TAG=triton_dt_identity_test \
  $REPODIR/build.sh
if [ -z $CUDA_VISIBLE_DEVICES ]
then
  docker run -v "${REPODIR}/qa/logs:/qa/logs" --gpus all --rm triton_dt_identity_test
else
  docker run -v "${REPODIR}/qa/logs:/qa/logs" --gpus $CUDA_VISIBLE_DEVICES --rm triton_dt_identity_test
fi
EXAMPLE_TAG=triton_dt_identity:cpu \
  TEST_TAG=triton_dt_identity_test:cpu \
  $REPODIR/build.sh --cpu-only
docker run -v "${REPODIR}/qa/logs:/qa/logs" --gpus all --rm triton_dt_identity_test:cpu
