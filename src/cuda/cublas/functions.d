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
 * Higher-level wrappers around CUDA cuBLAS.
 *
 * This module allows to use CUDA cuBLAS calls in a D-style using features like templates and error checking through asserts.
 */
module cuda.cublas.functions;

import cuda.common;
import cuda.cublas.types;
static import cublas = cuda.cublas.exp;

void cublasCreate(ref cublasHandle_t handle) nothrow @nogc
{
	enforceCublas(cublas.cublasCreate(&handle));
}

package void enforceCublas(cublasStatus_t error) pure nothrow @safe @nogc
{
	assert (error == cublasStatus_t.CUBLAS_STATUS_SUCCESS, error.toString);
}

