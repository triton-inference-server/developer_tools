/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/* Triton expects certain definitions within its backend libraries to follow
 * specific naming conventions. Specifically, for a backend named
 * "dev_tools_identity," most definitions should appear within a namespace called
 * triton::backend::dev_tools_identity.
 *
 * In order to facilitate this with minimal effort on the part of backend
 * developers, we ask that you put the name of your backend here. This macro is
 * then used to propagate the correct namespace name wherever it is needed in
 * the impl and interface code. */

#define NAMESPACE dev_tools_identity
