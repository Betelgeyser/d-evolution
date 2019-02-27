/**
 * Copyright © 2018 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
 * CUDA cuBLAS exported functions.
 */
module cuda.cublas.exp;

import cuda.cublas.types;

package:
	alias cublasCreate  = cublasCreate_v2;
	alias cublasDestroy = cublasDestroy_v2;
	alias cublasSgemm   = cublasSgemm_v2;

extern(C) nothrow pure @nogc:
	cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
	cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
	cublasStatus_t cublasSgemm_v2(
		cublasHandle_t handle,
		cublasOperation_t transa, cublasOperation_t transb,
		int m, int n, int k,
		const(float)* alpha,
		const(float)* A, int lda,
		const(float)* B, int ldb,
		const(float)* beta,
		float* C, int ldc
	);
	cublasStatus_t cublasSgeam(
		cublasHandle_t handle,
		cublasOperation_t transa, cublasOperation_t transb,
		int m, int n,
		const(float)* alpha,
		const(float)* A, int lda,
		const(float)* beta,
		const(float)* B, int ldb,
		float* C, int ldc
	);

