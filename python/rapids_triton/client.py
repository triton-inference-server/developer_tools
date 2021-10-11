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

import time

from rapids_triton.triton.client import get_triton_client
from rapids_triton.triton.io import create_triton_input, create_triton_output

from rapids_triton.triton.dtype import dtype_to_triton_name
from rapids_triton.triton.response import get_response_data
from tritonclient import utils as triton_utils


class Client(object):
    def __init__(
            self,
            protocol='grpc',
            host='localhost',
            port=None,
            concurrency=4):
        self.triton_client = get_triton_client(
            protocol=protocol,
            host=host,
            port=port,
            concurrency=concurrency
        )
        self._protocol = protocol

    @property
    def protocol(self):
        return self._protocol

    def create_input(self, data, name, dtype, shared_mem=None):
        return create_triton_input(
            self.triton_client,
            data,
            name,
            dtype,
            protocol=self.protocol,
            shared_mem=shared_mem
        )

    def create_output(self, size, name, shared_mem=None):
        return create_triton_output(
            self.triton_client,
            size,
            name,
            protocol=self.protocol,
            shared_mem=shared_mem
        )

    def wait_for_server(self, timeout):
        server_wait_start = time.time()
        while True:
            try:
                if self.triton_client.is_server_ready():
                    break
            except triton_utils.InferenceServerException:
                pass
            if time.time() - server_wait_start > timeout:
                raise RuntimeError("Server startup timeout expired")
            time.sleep(1)

    def clear_shared_memory(self):
        self.triton_client.unregister_cuda_shared_memory()
        self.triton_client.unregister_system_shared_memory()

    def get_model_config(self, model_name):
        return self.triton_client.get_model_config(model_name).config

    def predict(
            self,
            model_name,
            input_data,
            output_sizes,
            model_version='1',
            shared_mem=None,
            attempts=1):
        model_version = str(model_version)

        try:
            inputs = [
                self.create_input(
                    arr,
                    name,
                    dtype_to_triton_name(arr.dtype),
                    shared_mem=shared_mem
                )
                for name, arr in input_data.items()
            ]

            outputs = {
                name: self.create_output(size, name, shared_mem=shared_mem)
                for name, size in output_sizes.items()
            }

            response = self.triton_client.infer(
                model_name,
                model_version=model_version,
                inputs=[input_.input for input_ in inputs],
                outputs=[output_.output for output_ in outputs.values()]
            )

        except triton_utils.InferenceServerException:
            if attempts > 1:
                return self.predict(
                    model_name,
                    input_data,
                    output_sizes,
                    model_version=model_version,
                    shared_mem=shared_mem,
                    attempts=attempts - 1
                )
            raise
        result = {
            name: get_response_data(response, handle, name)
            for name, (_, handle, _) in outputs.items()
        }

        for input_ in inputs:
            if input_.name is not None:
                self.triton_client.unregister_cuda_shared_memory(
                    name=input_.name
                )
        for output_ in outputs.values():
            if output_.name is not None:
                self.triton_client.unregister_cuda_shared_memory(
                    name=output_.name
                )
        return result
