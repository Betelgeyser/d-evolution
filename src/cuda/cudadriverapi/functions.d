/**
 * Copyright © 2018 Sergei Iurevich Filippov, All Rights Reserved.
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
 *
 * Higher-level wrappers around CUDA driver API.
 *
 * This module allows to use CUDA driver API calls in a D-style using features like templates and error checking through
 * asserts.
 */
module cuda.cudadriverapi.functions;

import cuda.common;
import cuda.cudadriverapi.types;
static import cudadriver = cuda.cudadriverapi.exp;

void cuMemsetD32(T)(T* dstDevice, uint ui, size_t N) nothrow @nogc
{
	enforceCuda(cudadriver.cuMemsetD32(dstDevice, ui, N));
}

/**
 * Utility wrapper to enforce error check for cuda functions.
 */
package void enforceCuda(CUresult error) pure nothrow @safe @nogc
{
	assert (error == CUresult.CUDA_SUCCESS, error.toString);
}

