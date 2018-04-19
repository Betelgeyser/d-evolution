/**
 * Copyright © 2017 - 2018 Sergei Iurevich Filippov, All Rights Reserved.
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
module math.matrix;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common;

version (unittest)
{
	import std.math : approxEqual;
	immutable accuracy = 0.000_001;
	
	cublasHandle_t cublasHandle;
	
	static this()
	{
		cublasCreate(cublasHandle);
	}
	
	static ~this()
	{
		cublasDestroy(cublasHandle);
	}
}


/**
 * Convenient struct to handle cuBLAS matricies.
 *
 * It wraps pointer to allocated memory of values with some additional properties of a matrix, such as rows and columns
 * numbers. All linear algebra method are implemented on GPU.
 *
 * As it implements cuBLAS matrix, it is column-major ordered.
 */
struct Matrix
{
	alias values this;
	
	float* values; /// A pointer to an allocated memory.
	uint   rows;   /// Rows number.
	uint   cols;   /// Columns number.
	
	/**
	 * The length of the matrix.
	 *
	 * Returns:
	 *     Number of elements.
	 */
	@property uint length() const pure nothrow @safe @nogc
	{
		return rows * cols;
	}
	
	invariant
	{
		assert (rows >= 1, "Matrix must containg at least 1 row.");
		assert (cols >= 1, "Matrix must containg at least 1 column.");
	}
	
	/**
	 * Create a matrix and allocate memory for it.
	 *
	 * Default values are not initialized. If a cuRAND generator is passed,
	 * values are randomly generated on GPU.
	 *
	 * Params:
	 *     rows = Number of rows.
	 *     cols = Number of columns.
	 *     generator = Pseudorandom number generator.
	 */
	this(in uint rows, in uint cols) nothrow @nogc
	in
	{
		assert (rows >= 1, "Matrix must containg at least 1 row.");
		assert (cols >= 1, "Matrix must containg at least 1 column.");
	}
	body
	{
		scope(failure) freeMem();
		
		this.rows = rows;
		this.cols = cols;
		
		cudaMallocManaged(values, length);
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		auto m = Matrix(3, 2);
		scope(exit) m.freeMem();
		
		cudaDeviceSynchronize();
		
		assert (m.rows == 3);
		assert (m.cols == 2);
		
		// Check memory accessebility
		assert (m.values[0] == m.values[0]);
		assert (m.values[m.length - 1] == m.values[m.length - 1]);
	}
	
	/**
	 * Free memory.
	 *
	 * For the reason how D works with structs memory freeing moved from destructor to
	 * the the distinct function. Either allocating structs on stack or in heap or both
	 * causes spontaneous destructors calls. Apparently structs are not intended
	 * to be used with dynamic memory, probably it should be implemented as a class.  
	 */
	void freeMem() nothrow @nogc
	{
		cudaFree(values);
	}
}

/**
 * Simpified version of gemm function.
 *
 * Performs matrix multiplication C = A * B.
 *
 * Params:
 *     A = The first matrix.
 *     B = The second matrix.
 *     C = Output matrix.
 *     cublasHandle = Cublas handle.
 */
void gemm(in Matrix A, in Matrix B, ref Matrix C, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (A.cols == B.rows);
	assert (A.rows == C.rows);
	assert (B.cols <= C.cols);
}
body
{
	immutable float alpha = 1;
	immutable float beta  = 0;
	
	cublasSgemm(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
		A.rows, B.cols, A.cols,
		&alpha,
		A, A.rows,
		B, B.rows,
		&beta,
		C, C.rows
	);
}

///
unittest
{
	mixin(writetest!gemm);
	
	immutable n = 7;
	immutable k = 3;
	immutable m = 5;
	
	auto A = Matrix(n, k);
	scope(exit) A.freeMem();
	
	auto B = Matrix(k, m);
	scope(exit) B.freeMem();
	
	auto C = Matrix(n, m);
	scope(exit) C.freeMem();
	
	for (ulong i = 0; i < A.length; ++i)
		A[i] = i;
	
	for (ulong i = 0; i < B.length; ++i)
		B[i] = i;
	
	gemm(A, B, C, cublasHandle);
	cudaDeviceSynchronize();
	
	// cuBLAS is column-major
	immutable float[] result = [
		35,  38,  41,  44,  47,  50,  53,
		98,  110, 122, 134, 146, 158, 170,
		161, 182, 203, 224, 245, 266, 287,
		224, 254, 284, 314, 344, 374, 404,
		287, 326, 365, 404, 443, 482, 521
	];
	for (ulong i = 0; i < C.length; ++i)
		assert ( approxEqual(C[i], result[i], accuracy) );
}

/**
 * Simpified version of geam function.
 *
 * Performs matrix addition. C = αA + βB.
 *
 * Params:
 *     alpha = α constant multiplier.
 *     A = The first matrix.
 *     beta = β constant multiplier.
 *     B = The second matrix.
 *     C = Output matrix.
 *     cublasHandle = Cublas handle.
 */
void geam(in float alpha, in Matrix A, in float beta, in Matrix B, ref Matrix C, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert ( (A.cols == B.cols) && (A.cols == C.cols) );
	assert ( (A.rows == B.rows) && (A.rows == C.rows) );
}
body
{
	cublasSgeam(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
		C.rows, C.cols,
		&alpha,
		A, A.rows,
		&beta,
		B, B.rows,
		C, C.rows
	);
}

///
unittest
{
	mixin(writetest!geam);
	
	immutable size = 10;
	
	auto A = Matrix(size, size);
	scope(exit) A.freeMem();
	
	auto B = Matrix(size, size);
	scope(exit) B.freeMem();
	
	auto C = Matrix(size, size);
	scope(exit) C.freeMem();
	
	for (ulong i = 0; i < A.length; ++i)
		A[i] = i;
	
	for (ulong i = 0; i < B.length; ++i)
		B[i] = i;
	
	geam(1, A, 2, B, C, cublasHandle);
	cudaDeviceSynchronize();
	
	for (ulong i = 0; i < C.length; ++i)
		assert ( approxEqual(C[i], 3 * i, accuracy) );
}

/**
 * Simpified version of geam function.
 *
 * Performs matrix transposition.
 *
 * Params:
 *     A = Matrix to transponse.
 *     C = Resulting matrix.
 *     cublasHandle = Cublas handle.
 */
void transpose(in Matrix A, ref Matrix C, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (A.rows == C.cols);
	assert (A.cols == C.rows);
}
body
{
	immutable float alpha = 1;
	immutable float beta  = 0;
	
	cublasSgeam(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_T, cublasOperation_t.CUBLAS_OP_N,
		A.cols, A.rows,
		&alpha,
		A, A.rows,
		&beta,
		A, A.rows,
		C, C.rows
	);
}

///
unittest
{
	mixin(writetest!transpose);
	
	immutable m = 5;
	immutable n = 3;
	
	auto A = Matrix(m, n);
	scope(exit) A.freeMem();
	
	auto C = Matrix(n, m);
	scope(exit) C.freeMem();
	
	for (ulong i = 0; i < A.length; ++i)
		A[i] = i;
	
	transpose(A, C, cublasHandle);
	cudaDeviceSynchronize();
	
	// cuBLAS is column-major
	immutable float[] result = [
		0, 5, 10,
		1, 6, 11,
		2, 7, 12,
		3, 8, 13,
		4, 9, 14
	];
	for (ulong i = 0; i < C.length; ++i)
		assert ( approxEqual(C[i], result[i], accuracy) );
}

