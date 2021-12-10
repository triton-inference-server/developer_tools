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

from collections import namedtuple
from uuid import uuid4

import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
from rapids_triton.utils.safe_import import ImportReplacement
from tritonclient import utils as triton_utils
try:
    import tritonclient.utils.cuda_shared_memory as shm
except OSError:  # CUDA libraries not available
    shm = ImportReplacement('tritonclient.utils.cuda_shared_memory')


TritonInput = namedtuple('TritonInput', ('name', 'input'))
TritonOutput = namedtuple('TritonOutput', ('name', 'handle', 'output'))

def set_unshared_input_data(triton_input, data, protocol='grpc'):
    if protocol == 'grpc':
        triton_input.set_data_from_numpy(data)
    else:
        triton_input.set_data_from_numpy(data, binary_data=True)

    return None


def set_shared_input_data(triton_client, triton_input, data, protocol='grpc'):
    input_size = data.size * data.itemsize

    input_name = 'input_{}'.format(uuid4().hex)

    input_handle = shm.create_shared_memory_region(
        input_name, input_size, 0
    )

    shm.set_shared_memory_region(input_handle, [data])

    triton_client.register_cuda_shared_memory(
        input_name, shm.get_raw_handle(input_handle), 0, input_size
    )

    triton_input.set_shared_memory(input_name, input_size)

    return input_name


def set_input_data(
        triton_client,
        triton_input,
        data,
        protocol='grpc',
        shared_mem=None):
    if shared_mem is None:
        return set_unshared_input_data(
            triton_input, data, protocol=protocol
        )
    if shared_mem == 'cuda':
        return set_shared_input_data(
            triton_client, triton_input, data, protocol=protocol
        )
    raise RuntimeError("Unsupported shared memory type")


def create_triton_input(
        triton_client, data, name, dtype, protocol='grpc', shared_mem=None):
    if protocol == 'grpc':
        triton_input = triton_grpc.InferInput(name, data.shape, dtype)
    else:
        triton_input = triton_http.InferInput(name, data.shape, dtype)

    input_name = set_input_data(
        triton_client,
        triton_input,
        data,
        protocol=protocol,
        shared_mem=shared_mem
    )

    return TritonInput(name=input_name, input=triton_input)


def create_output_handle(triton_client, triton_output, size, shared_mem=None):
    if shared_mem is None:
        return (None, None)

    output_name = 'output_{}'.format(uuid4().hex)
    output_handle = shm.create_shared_memory_region(
        output_name, size, 0
    )

    triton_client.register_cuda_shared_memory(
        output_name, shm.get_raw_handle(output_handle), 0, size
    )

    triton_output.set_shared_memory(output_name, size)

    return output_name, output_handle


def create_triton_output(
        triton_client, size, name, protocol='grpc', shared_mem=None):
    """Set up output memory in Triton

    Parameters
    ----------
    triton_client : Triton client object
        The client used to set output parameters
    size : int
        The size of the output in bytes
    name : str
        The model-defined name for this output
    protocol : 'grpc' or 'http'
        The protocol used for communication with the server
    """
    if protocol == 'grpc':
        triton_output = triton_grpc.InferRequestedOutput(name)
    else:
        triton_output = triton_grpc.InferRequestedOutput(
            name, binary_data=True
        )

    output_name, output_handle = create_output_handle(
        triton_client, triton_output, size, shared_mem=shared_mem
    )

    return TritonOutput(
        name=output_name,
        handle=output_handle,
        output=triton_output
    )
