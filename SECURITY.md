<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Security Policy

## Reporting a Vulnerability

Please do **not** report security vulnerabilities through public GitHub
issues, discussions, or pull requests.

To report a potential security vulnerability in any NVIDIA product, use one of
the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   <https://www.nvidia.com/en-us/security/>
2. **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt
   sensitive reports with NVIDIA's public PGP key:
   <https://www.nvidia.com/en-us/security/pgp-key>
3. **GitHub Private Vulnerability Reporting:** use the "Report a
   vulnerability" button on this repository's **Security** tab, where enabled.

Please include:

1. Product name and version, branch, or commit that contains the
   vulnerability.
2. Type of vulnerability (for example code execution, denial of service,
   buffer overflow, path traversal).
3. Step-by-step instructions to reproduce the issue.
4. Proof-of-concept or exploit code, if available.
5. Potential impact, including how an attacker could exploit the issue.

NVIDIA PSIRT acknowledges reports, assesses them, and coordinates fixes and
disclosure with the reporter. OEM partners should contact their NVIDIA
Customer Program Manager. Past bulletins are listed at
<https://www.nvidia.com/en-us/security/>.

## Security Architecture and Context

**Project:** Triton Developer Tools, a set of helper libraries and tooling
for building applications on top of the Triton Inference Server.

**Classification:** Library / SDK, plus build and development tooling. It is
not a network service and does not open listening sockets itself.

**Components:**

- `server/` is a C++17 wrapper library (`TritonServer`, `InferRequest`,
  `InferResult`, and a trace manager) over the in-process Triton Server C API
  (`libtritonserver`), with example programs and a unit test.
- `tools/add_copyright.py` and `.pre-commit-hooks.yaml` provide the
  `add-license` pre-commit hook used by Triton repositories. It reads and
  rewrites source files in the repository where it runs.
- `server/install_dependencies_and_build.sh` installs build dependencies and
  builds the wrapper.
- `qa/` contains test scripts and sample Python model fixtures.

**Primary security responsibility:** correct handling of caller-supplied model
repository paths, tensor buffers, memory types, and trace file paths when
passing them to the Triton core library, and safe, bounded file modification
by the license hook.

**Key boundaries and interfaces:**

- The public C++ API in `server/include/triton/developer_tools/`.
- Pre-commit hook invocation with file paths supplied by pre-commit.
- Build-time package installation and downloads.

**Repository Exposure Classification:** Public. Basis: the repository is
publicly visible on GitHub.

**Service Exposure Classification:** Internal-Isolated, medium confidence.
Basis: the code is an embedded library and developer tool with no network
listeners and no authentication surface of its own. Exposure is inherited
from the application that embeds it.

## Threat Model

1. **Untrusted model repository or model content:** `TritonServer::Create`
   accepts a model repository path from the embedding application. A
   repository containing a malicious model or backend library can execute
   code in the host process, because models and backends are loaded
   in-process.
2. **Unsafe buffer and memory handling in the tensor API:** `InferRequest`
   and `Tensor` accept caller-provided buffers, sizes, data types, and memory
   types (CPU, pinned, GPU). Mismatched sizes or types supplied by a caller
   can lead to out-of-bounds reads or writes in the C++ wrapper.
3. **Trace file path abuse:** the trace manager in `server/src/tracer.cc`
   writes trace output to caller-supplied file paths. An attacker who
   controls that path or its location may overwrite files or cause
   unbounded disk usage through a high trace rate.
4. **Unintended file modification by the license hook:**
   `tools/add_copyright.py` rewrites files in place. Running it on untrusted
   or unexpected paths, or through symlinks, can modify files outside the
   intended repository.
5. **Build-time supply chain:** `server/install_dependencies_and_build.sh`
   adds an external package repository and signing key and installs a pinned
   CMake version. Compromise of that source or of the build base image would
   affect built artifacts.
6. **Pre-commit consumers pinned by tag:** downstream repositories consume
   the `add-license` hook by tag. Replacing or moving a tag would change code
   executed on every contributor's machine and in CI.

## Critical Security Assumptions

- The embedding application is trusted and is responsible for authenticating
  and authorizing its own users. This library provides no authentication,
  authorization, or TLS.
- Model repositories, backends, and shared libraries loaded by Triton come
  from trusted sources. The library does not verify their integrity.
- Callers validate tensor shapes, sizes, data types, and memory types before
  passing them to the wrapper.
- Trace file paths are chosen by the application operator and are not
  attacker-controlled. The destination has sufficient disk space and
  appropriate permissions.
- The license hook runs only against files inside a trusted working tree.
- Build environments, base images, and package mirrors used for
  `install_dependencies_and_build.sh` are trusted.
- Release tags are write-once and are not moved after publication.
