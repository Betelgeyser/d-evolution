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
 */
module cuda.cublas.types;

/**
 * The cublasHandle_t type is a pointer type to an opaque structure holding the cuBLAS library context
 *
 * Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz53yDMI41e
 */
alias cublasHandle_t = cublasContext*;
private struct cublasContext;

/**
 * cuBLAS statucs.
 */
enum cublasStatus_t
{
	CUBLAS_STATUS_SUCCESS          = 0,
	CUBLAS_STATUS_NOT_INITIALIZED  = 1,
	CUBLAS_STATUS_ALLOC_FAILED     = 3,
	CUBLAS_STATUS_INVALID_VALUE    = 7,
	CUBLAS_STATUS_ARCH_MISMATCH    = 8,
	CUBLAS_STATUS_MAPPING_ERROR    = 11,
	CUBLAS_STATUS_EXECUTION_FAILED = 13,
	CUBLAS_STATUS_INTERNAL_ERROR   = 14,
	CUBLAS_STATUS_NOT_SUPPORTED    = 15,
	CUBLAS_STATUS_LICENSE_ERROR    = 16
}

/**
 * cuBLAS matrix operation type.
 *
 * Read more at: http://docs.nvidia.com/cuda/cublas/index.html#ixzz53yBlVPU6
 */
enum cublasOperation_t
{
	CUBLAS_OP_N = 0, /// The non-transpose operation
	CUBLAS_OP_T = 1, /// The transpose operation
	CUBLAS_OP_C = 2  /// The conjugate transpose operation
}

