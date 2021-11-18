###########################################################################################
# Arguments for controlling build details
###########################################################################################
# Version of Triton to use
ARG TRITON_VERSION=21.10
# Base container image
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
# Whether or not to build indicated components
ARG BUILD_TESTS=OFF
ARG BUILD_EXAMPLE=ON
# Whether or not to enable GPU build
ARG TRITON_ENABLE_GPU=ON

FROM ${BASE_IMAGE} as base

ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y wget patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=true

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

COPY ./conda/environments/rapids_triton_dev.yml /environment.yml

RUN conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete

ENV PYTHONDONTWRITEBYTECODE=false

SHELL ["conda", "run", "--no-capture-output", "-n", "rapids_triton_dev", "/bin/bash", "-c"]

FROM base as build-stage

COPY ./cpp /rapids_triton

ARG TRITON_VERSION
ENV TRITON_VERSION=$TRITON_VERSION

ARG BUILD_TYPE=Release
ENV BUILD_TYPE=$BUILD_TYPE
ARG BUILD_TESTS
ENV BUILD_TESTS=$BUILD_TESTS
ARG BUILD_EXAMPLE
ENV BUILD_EXAMPLE=$BUILD_EXAMPLE
ARG TRITON_ENABLE_GPU
ENV TRITON_ENABLE_GPU=$TRITON_ENABLE_GPU

RUN mkdir /rapids_triton/build

WORKDIR /rapids_triton/build

RUN cmake \
      -GNinja \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DBUILD_TESTS="${BUILD_TESTS}" \
      -DBUILD_EXAMPLE="${BUILD_EXAMPLE}" \
      -DTRITON_ENABLE_GPU="${TRITON_ENABLE_GPU}" \
      ..

ENV CCACHE_DIR=/ccache

RUN --mount=type=cache,target=/ccache/ ninja install && ccache -s

FROM base as test-install

COPY ./conda/environments/rapids_triton_test.yml /environment.yml

RUN conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete

FROM build-stage as test-stage

COPY --from=test-install /root/miniconda3 /root/miniconda3

ENV TEST_EXE=/rapids_triton/build/test_rapids_triton

COPY qa /qa

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rapids_triton_test", "/bin/bash", "/qa/entrypoint.sh"]

FROM ${BASE_IMAGE}

RUN mkdir /models

# Remove existing backend install
RUN if [ -d /opt/tritonserver/backends/rapids-identity ]; \
    then \
      rm -rf /opt/tritonserver/backends/rapids-identity/*; \
    fi

COPY --from=build-stage \
  /opt/tritonserver/backends/rapids-identity \
  /opt/tritonserver/backends/rapids-identity

ENTRYPOINT ["tritonserver", "--model-repository=/models"]
