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

void cublasDestroy(cublasHandle_t handle) nothrow @nogc
{
	enforceCublas(cublas.cublasDestroy(handle));
}

/**
 * This function performs the matrix-matrix multiplication:
 *
 * C = αop(A)op(B) + βC
 *
 * where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A) m × k,
 * op(B) k × n and C m × n, respectively. Also, for matrix A op(A) = A if transa == CUBLAS_OP_N A T,
 * if transa == CUBLAS_OP_T A H, if transa == CUBLAS_OP_C and op(B) is defined similarly for matrix B.
 *
 * Params:
 *     handle = handle to the cuBLAS library context.
 *     transa = operation op(A) that is non- or (conj.) transpose.
 *     transb = operation op(B) that is non- or (conj.) transpose.
 *     m = number of rows of matrix op(A) and C.
 *     n = number of columns of matrix op(B) and C.
 *     k = number of columns of op(A) and rows of op(B).
 *     alpha = scalar used for multiplication.
 *     A = array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
 *     lda = leading dimension of two-dimensional array used to store the matrix A.
 *     B = array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
 *     ldb = leading dimension of two-dimensional array used to store matrix B.
 *     beta = scalar used for multiplication. If beta==0, C does not have to be a valid input.
 *     C = array of dimensions ldc x n with ldc>=max(1,m).
 *     ldc = leading dimension of a two-dimensional array used to store the matrix C.
 */
void cublasSgemm(
	cublasHandle_t handle,
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const(float)* alpha,
	const(float)* A, int lda,
	const(float)* B, int ldb,
	const(float)* beta,
	float* C, int ldc
) nothrow @nogc
{
	enforceCublas(
		cublas.cublasSgemm(
			handle,
			transa, transb,
			m, n, k,
			alpha,
			A, lda,
			B, ldb,
			beta,
			C, ldc
		)
	);
}

/**
 * This function performs the matrix-matrix addition/transposition:
 *
 * C = αop(A) + βop(B)
 *
 * where α and β are scalars, and A, B and C are matrices stored in column-major format with dimensions op(A) m × n,
 * op(B) m × n and C m × n, respectively. Also, for matrix A op(A) = A, if transa == CUBLAS_OP_N A T,
 * if transa == CUBLAS_OP_T A H, if transa == CUBLAS_OP_C and op(B) is defined similarly for matrix B.
 *
 * The operation is out-of-place if C does not overlap A or B.
 *
 * The in-place mode supports the following two operations,
 * C = α * C + βop(B)
 * C = αop(A) + β * C
 * For in-place mode, if C = A, ldc = lda and transa = CUBLAS_OP_N. If C = B, ldc = ldb and transb = CUBLAS_OP_N.
 * If the user does not meet above requirements, CUBLAS_STATUS_INVALID_VALUE is returned.
 *
 * The operation includes the following special cases:
 * the user can reset matrix C to zero by setting *alpha=*beta=0.
 * the user can transpose matrix A by setting *alpha=1 and *beta=0.
 *
 * Params:
 *     handle = handle to the cuBLAS library context.
 *     transa = operation op(A) that is non- or (conj.) transpose.
 *     transb = operation op(B) that is non- or (conj.) transpose.
 *     m = number of rows of matrix op(A) and C.
 *     n = number of columns of matrix op(B) and C.
 *     alpha = scalar used for multiplication. If *alpha == 0, A does not have to be a valid input.
 *     A = array of dimensions lda x n with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,n) otherwise.
 *     lda = leading dimension of two-dimensional array used to store the matrix A.
 *     B = array of dimension ldb x n with ldb>=max(1,m) if transa == CUBLAS_OP_N and ldb x m with ldb>=max(1,n) otherwise.
 *     ldb = leading dimension of two-dimensional array used to store matrix B.
 *     beta = scalar used for multiplication. If *beta == 0, B does not have to be a valid input.
 *     C = array of dimensions ldc x n with ldc>=max(1,m).
 *     ldc = leading dimension of a two-dimensional array used to store the matrix C.
 */
void cublasSgeam(
	cublasHandle_t handle,
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n,
	const(float)* alpha,
	const(float)* A, int lda,
	const(float)* beta,
	const(float)* B, int ldb,
	float* C, int ldc
) nothrow @nogc
{
	enforceCublas(
		cublas.cublasSgeam(
			handle,
			transa, transb,
			m, n,
			alpha,
			A, lda,
			beta,
			B, ldb,
			C, ldc
		)
	);
}

package void enforceCublas(cublasStatus_t error) pure nothrow @safe @nogc
{
	assert (error == cublasStatus_t.CUBLAS_STATUS_SUCCESS, error.toString);
}

