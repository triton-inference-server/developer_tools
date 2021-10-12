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

import numpy as np
import pytest

from rapids_triton import Client
from rapids_triton.testing import get_random_seed, arrays_close

TOTAL_SAMPLES = 8192
FEATURE_COUNT = 4
ALPHA = 2
C = np.array([[1, 2, 3, 4]], dtype='float32')

@pytest.fixture
def model_inputs():
    np.random.seed(get_random_seed())
    return {
        input_name:
        np.random.rand(TOTAL_SAMPLES, FEATURE_COUNT).astype('float32')
        for input_name in ('u', 'v')
    }

@pytest.fixture
def model_output_sizes():
    return {'r': TOTAL_SAMPLES * FEATURE_COUNT * np.dtype('float32').itemsize}

def get_ground_truth(inputs):
    u = inputs['u']
    v = inputs['v']
    return {'r': ALPHA * u + v + np.repeat(C, u.shape[0], axis=0)}


@pytest.mark.parametrize(
    "model_name", ['linear_example', 'linear_example_cpu']
)
def test_model(model_name, model_inputs, model_output_sizes):
    client = Client()
    result = client.predict(model_name, model_inputs, model_output_sizes)
    # shm_result = client.predict(
    #     model_name, model_inputs, model_output_sizes, shared_mem='cuda'
    # )
    ground_truth = get_ground_truth(model_inputs)

    for output_name in sorted(ground_truth.keys()):
        arrays_close(
            result[output_name],
            ground_truth[output_name],
            atol=1e-5,
            assert_close=True
        )
        # arrays_close(
        #     shm_result[output_name],
        #     ground_truth[output_name],
        #     atol=1e-5,
        #     assert_close=True
        # )
