###########################################################################################
# Arguments for controlling build details
###########################################################################################
# Version of Triton to use
ARG TRITON_VERSION=21.09
# Base container image
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
# Whether or not to build indicated components
ARG BUILD_TESTS=OFF
ARG BUILD_EXAMPLE=ON

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

COPY ./conda/environments/rapids_triton_dev_cuda11.4.yml /environment.yml

RUN conda env update -f /environment.yml \
    && rm /environment.yml \
    && conda clean -afy \
    && find /root/miniconda3/ -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda3/ -follow -type f -name '*.js.map' -delete

ENV PYTHONDONTWRITEBYTECODE=false

RUN mkdir /rapids_triton

COPY ./src /rapids_triton/src
COPY ./CMakeLists.txt /rapids_triton
COPY ./cmake /rapids_triton/cmake

WORKDIR /rapids_triton

SHELL ["conda", "run", "--no-capture-output", "-n", "rapids_triton_dev", "/bin/bash", "-c"]

FROM base as build-stage

ARG TRITON_VERSION
ENV TRITON_VERSION=$TRITON_VERSION

ARG BUILD_TYPE=Release
ENV BUILD_TYPE=$BUILD_TYPE
ARG BUILD_TESTS
ENV BUILD_TESTS=$BUILD_TESTS
ARG BUILD_EXAMPLE
ENV BUILD_EXAMPLE=$BUILD_EXAMPLE

RUN mkdir /rapids_triton/build /rapids_triton/install

WORKDIR /rapids_triton/build

RUN cmake \
      -GNinja \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DBUILD_TESTS="${BUILD_TESTS}" \
      -DCMAKE_INSTALL_PREFIX=/rapids_triton/install \
      -DTRITON_COMMON_REPO_TAG="r${TRITON_VERSION}" \
      -DTRITON_CORE_REPO_TAG="r${TRITON_VERSION}" \
      -DTRITON_BACKEND_REPO_TAG="r${TRITON_VERSION}" \
      ..

RUN ninja install

FROM ${BASE_IMAGE}

ARG BACKEND_NAME
ENV BACKEND_NAME=$BACKEND_NAME

RUN mkdir /models

# Remove existing backend install
RUN if [ -d /opt/tritonserver/backends/${BACKEND_NAME} ]; \
    then \
      rm -rf /opt/tritonserver/backends/${BACKEND_NAME}/*; \
    fi

COPY --from=build-stage \
  /opt/tritonserver/backends/$BACKEND_NAME \
  /opt/tritonserver/backends/$BACKEND_NAME

ENTRYPOINT ["tritonserver", "--model-repository=/models"]
