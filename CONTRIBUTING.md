<!--
Copyright (c) 2021, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!-- TODO(wphicks): Add more detail -->
# Contributing to RAPIDS-Triton

You can help improve RAPIDS-Triton in any of the following ways:
- Submitting a bug report, feature request or documentation issue
- Proposing and implementing a new feature
- Implementing a feature or bug-fix for an outstanding issue

## Bug reports
When submitting a bug report, please include a *minimum* *reproducible*
example. Ideally, this should be a snippet of code that other developers can
copy, paste, and immediately run to try to reproduce the error. Please:
- Do include import statements and any other code necessary to immediately run
  your example
- Avoid examples that require other developers to download models or data
  unless you cannot reproduce the problem with synthetically-generated data

## Code Contributions
To contribute code to this project, please follow these steps:
1. Find an issue to work on or submit an issue documenting the problem you
   would like to work on.
2. Comment on the issue saying that you plan to work on it.
3. Review the conventions below for information to help you make your changes
   in a way that is consistent with the rest of the codebase.
4. Code!
5. Create your pull request.
6. Wait for other developers to review your code and update your PR as needed.
7. Once a PR is approved, it will be merged into the main branch.

### Coding Conventions
* RAPIDS-Triton follows [Almost Always Auto
  (AAA)](https://herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/)
  style. Please maintain this style in any contributions, with the possible
  exception of some docs, where type information may be helpful for new users
  trying to understand a snippet in isolation.
* Avoid raw loops where possible.
* C++ versions of types should be used instead of C versions except when
  interfacing with C code (e.g. use `std::size_t` instead of `size_t`).
* Avoid using output pointers in function signatures. Prefer instead to
  actually return the value computed by the function and take advantage of
  return value optimization and move semantics.

### Signing Your Work
* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
  * Any contribution which contains commits that are not Signed-Off will not be accepted.
* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```
* Full text of the DCO:
  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  
  
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
