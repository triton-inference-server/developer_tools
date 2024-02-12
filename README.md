# Triton Developer Tools
This repository contains tools to make it easier to develop custom C++ backends
for NVIDIA's Triton Inference Server and to use Triton as a library within an
existing C++ project. It consists of two sub-projects:
- [Triton Backend Developer Tools](https://github.com/triton-inference-server/developer_tools/tree/main/backend): A header-only library to make it quick and easy to add high-performance custom functionality to Triton at the C++ level
- Triton Library Developer Tools: Coming soon, this library will allow easy
  integration of Triton as a library within other C++ projects

## Backend Developer Tools

### Why might I want to use Triton's Backend Developer Tools?
- You have an inference model which you wish to serve in Triton, but Triton
  does not currently support your model type
- You want to add specialized data manipulation logic to Triton and require
  performance beyond what the Python backend can provide
- You want to ensure that your custom C++ backend stays up-to-date with the
  latest Triton performance features

### Why might I _not_ want to use Triton's Backend Developer Tools?
- The functionality you're looking for is already provided by an existing
  Triton backend
- You want to write your additional logic in Python rather than C++

### Example
To create a custom backend with Triton's Backend Developer Tools, you can
make use of the Backend Template (TODO(wphicks): link) and have a working
Triton backend in minutes. For example, consider a simple example of a backend
that simply returns any array of floats provided as input. In order to create
such a backend, the only code we would have to write would be the
following `predict` method:
```
  void predict(dev_tools::Batch& batch) const {
    dev_tools::Tensor<float> input = get_input<float>(batch, "input__0");
    dev_tools::Tensor<float> output = get_output<float>(batch, "output__0");

    dev_tools::copy(output, input);

    output.finalize();
  }
```
For a complete walkthrough of this example and much more detail on how to add
your own functionality to Triton, check out the Backend Developer Tools docs
TODO(wphicks): Link.

## Library Developer Tools
TODO(wphicks)
