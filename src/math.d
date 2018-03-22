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
 *
 * This module contains basic structs, subroutines and cuda kernel interfaces for mathematics.
 */
module math;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common;

version (unittest)
{
	import std.math : approxEqual;
	immutable accuracy = 0.000001;
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
	
	/// ditto
	this(in uint rows, in uint cols, curandGenerator_t generator) nothrow @nogc
	in
	{
		assert (rows >= 1, "Matrix must containg at least 1 row.");
		assert (cols >= 1, "Matrix must containg at least 1 column.");
	}
	body
	{
		this(rows, cols);
		
		{
			// To prevent a double free error, additional freeMem() is moved from this(...) scope.
			scope(failure) freeMem();
			curandGenerate(generator, values, length);
		}
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		
		scope(exit) curandDestroyGenerator(generator);
		
		auto m = Matrix(3, 2, generator); scope(exit) m.freeMem();
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
 * Calculate an Absolute Error between $(D_PARAM A) and $(D_PARAM B) arrays of vectors on GPU.
 *
 * Resulting array $(D_PARAM error) is calculated by formula:
 *
 * <math>
 *   <mrow>
 *     <msub> <mi>error</mi> <mi>i</mi> </msub>
 *     <mo>=</mo>
 *     <msqrt> <msup>
 *         <mfenced open="(" close=")" separators="">
 *           <msub> <mi>A</mi> <mi>i</mi> </msub>
 *           <mo>-</mo>
 *           <msub> <mi>B</mi> <mi>i</mi> </msub>
 *         </mfenced>
 *         <mn>2</mn>
 *     </msup> </msqrt>
 *   </mrow>
 * </math>
 *
 * Though $(D_PARAM A) and $(D_PARAM B) are of the type `Matrix` this is a technical convinience. They are interpreted
 * as arrays of vectors where a single column is a single vector.
 *
 * Params:
 *     A = The first array of vectors.
 *     B = The second array of vectors.
 *     error = The resulting array of errors. 
 *     cublasHandle = cuBLAS handle.
 */
void AE(in Matrix A, in Matrix B, ref Matrix error, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (A.cols == B.cols);
	assert (A.rows == B.rows);
	assert (A.cols == error.cols);
	assert (error.rows == 1);
}
body
{
	float alpha =  1;
	float beta  = -1;
	
	auto C = Matrix(A.rows, A.cols);
	
	cublasSgeam(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
		A.rows, A.cols,
		&alpha,
		A.values, A.rows,
		&beta,
		B.values, B.rows,
		C, C.rows
	);
	
	cuda_L2(C, error, C.rows, error.cols);
}

///
unittest
{
	mixin(writetest!AE);
	
	cublasHandle_t handle;
	cublasCreate(handle);
	scope(exit) cublasDestroy(handle);
	
	auto A = Matrix(3, 4);
	auto B = Matrix(3, 4);
	auto E = Matrix(1, 4);
	
	for (ulong i = 0; i < A.length; ++i)
	{
		A[i] = i;
		B[i] = i * 1.5;
	}
	
	AE(A, B, E, handle);
	cudaDeviceSynchronize();
	
	float[] result = [1.118034, 3.535534, 6.103278, 8.689074];
	for (ulong i = 0; i < E.length; ++i)
		assert ( approxEqual(E[i], result[i], accuracy) );
}

/**
 * Calculate a Mean Absolute Error between $(D_PARAM A) and $(D_PARAM B) array of vectors on GPU.
 *
 * Though $(D_PARAM A) and $(D_PARAM B) are of the type `Matrix` this is a technical convinience. They are interpreted
 * as arrays of vectors where a single column is a single vector.
 *
 * Params:
 *     A = The first array of vectors.
 *     B = The second array of vectors.
 *     cublasHandle = cuBLAS handle.
 *
 * See_also:
 *     $(LINK https://en.wikipedia.org/wiki/Mean_absolute_error)
 */
float MAE(in Matrix A, in Matrix B, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (A.cols == B.cols);
	assert (A.rows == B.rows);
}
body
{
	float result = 0;
	
	auto error = Matrix(1, A.cols);
	
	AE(A, B, error, cublasHandle);
	cudaDeviceSynchronize();
	
	for (ulong i = 0; i < error.length; ++i)
		result += error[i] / error.length;
	
	return result;
}

///
unittest
{
	mixin(writetest!MAE);
	
	cublasHandle_t handle;
	cublasCreate(handle);
	scope(exit) cublasDestroy(handle);
	
	auto A = Matrix(3, 4);
	auto B = Matrix(3, 4);
	
	for (ulong i = 0; i < A.length; ++i)
	{
		A[i] = i;
		B[i] = i * 1.5;
	}
	
	assert ( approxEqual(MAE(A, B, handle), 4.861480, accuracy) );
}

/**
 * Calculate a Mean Absolute Error of naive forecast on GPU.
 *
 * Useful for MASE calculation.
 *
 * Though $(D_PARAM data) is of the type `Matrix` this is a technical convinience. It is interpreted as an array of vectors
 * where a single column is a single vector.
 *
 * Params:
 *     data = An array of input vectors.
 *     cublasHandle = cuBLAS handle.
 */
float MAEnaive(in Matrix data, cublasHandle_t cublasHandle) nothrow @nogc
{
	auto measured = Matrix(data.rows, data.cols - 1);
	measured.values = cast(float*)data.values; // Just pointers, no copying here
	
	auto naive = Matrix(data.rows, data.cols - 1);
	naive.values = cast(float*)data.values + data.rows; // Shift one column to the begining
	
	return MAE(measured, naive, cublasHandle);
}

///
unittest
{
	mixin(writetest!MAEnaive);
	
	cublasHandle_t handle;
	cublasCreate(handle);
	scope(exit) cublasDestroy(handle);
	
	auto data = Matrix(2, 3);
	for (ulong i = 0; i < data.length; ++i)
		data[i] = i * i;
	
	assert ( approxEqual(MAEnaive(data, handle), 14.472136, 0.000001) );
}

/**
 * Calculate a Mean Absolute Scalde Error between $(D_PARAM measured) and $(D_PARAM approximated) arrays of vectors on GPU.
 *
 * Though $(D_PARAM data) is of the type `Matrix` this is a technical convinience. It is interpreted as an array of vectors
 * where a single column is a single vector.
 *
 * Params:
 *     measured = An array of vectors of measured/actual/real data.
 *     approximated = An array of vectors of approximated/estimated data.
 *     cublasHandle = cuBLAS handle.
 */
float MASE(in Matrix measured, in Matrix approximated, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (measured.rows == approximated.rows);
	assert (measured.cols == approximated.cols);
}
body
{
	return MAE(measured, approximated, cublasHandle) / MAEnaive(measured, cublasHandle);
}

///
unittest
{
	mixin(writetest!MASE);
	
	cublasHandle_t handle;
	cublasCreate(handle);
	scope(exit) cublasDestroy(handle);
	
	auto measured = Matrix(3, 4);
	for (ulong i = 0; i < measured.length; ++i)
		measured[i] = i;
	
	auto approximated = Matrix(3, 4);
	for (ulong i = 0; i < approximated.length; ++i)
		approximated[i] = i + 1;
	
	assert (
		approxEqual(
			MASE(measured, approximated, handle),
			0.333333,
			accuracy
		)
	);
}

extern (C++):
	/**
	 * Calculate hyperbolic tangent for each element of an array `x` on GPU.
	 *
	 * Params:
	 *     x = A pointer to an array. Could point to not the first element.
	 *     n = Size of the array. If n is less than the actual x size, only the first n elements starting from the pointer p
	 *         will be calculated.
	 */
	void cuda_tanh(float* x, size_t n) nothrow @nogc;
	
	///
	unittest
	{
		mixin(writetest!cuda_tanh);
		
		immutable length = 5;
		
		float* data;
		cudaMallocManaged(data, length);
		data[0] = -1_000;
		data[1] =     -1;
		data[2] =      0;
		data[3] =      1;
		data[4] =  1_000;
		
		cuda_tanh(data, length);
		cudaDeviceSynchronize();
		
		immutable float[] result = [-1.000000, -0.761594, 0.000000,  0.761594, 1.000000];
		for (ulong i = 0; i < length; ++i)
			assert ( approxEqual(data[i], result[i], accuracy) );
	}
	
	/**
	 * Fill an array on GPU.
	 *
	 * Params:
	 *     x = A pointer to an array. Could point to not the first element.
	 *     val = A value to fill with.
	 *     n = Size of the array. If n is less than the actual x size, only the first n elements starting from the pointer p
	 *         will be filled.
	 */
	void cuda_fill(float* x, float val, size_t n) nothrow @nogc;
	
	///
	unittest
	{
		mixin(writetest!cuda_fill);
		
		immutable length = 5;
		
		float* data;
		cudaMallocManaged(data, length);
		
		cuda_fill(data,     1, length);
		cuda_fill(data + 1, 2, length - 2);
		cudaDeviceSynchronize();
		
		immutable float[] result = [1, 2, 2, 2, 1];
		for (ulong i = 0; i < length; ++i)
			assert ( approxEqual(data[i], result[i], accuracy) );
	}
	
	/**
	 * Per-vector calculation of the Euclidean distance (L2 norm) of a vector array on GPU.
	 *
	 * Params:
	 *     x = A pointer to an array of vectors. Must have size of `dim * count` or less but be multiple to `dim`.
	 *     y = A pointer to the resulting array of L2 norm values. Must contain `count` elements.
	 *     dim = Vectors dimention.
	 *     count = Number of vectors in the `x` array and resulting values in the `y` array.
	 */
	void cuda_L2(const(float)* x, float* y, int dim, size_t count) nothrow @nogc;
	
	///
	unittest
	{
		mixin(writetest!cuda_L2);
		
		immutable dim    = 4;
		immutable length = 2;
		
		float* data;
		float* norm;
		cudaMallocManaged(data, dim * length);
		cudaMallocManaged(norm, length);
		
		for (ulong i = 0; i < dim * length; ++i)
			data[i] = i;
		
		cuda_L2(data, norm, dim, length);
		cudaDeviceSynchronize();
		
		immutable float[] result = [3.741657, 11.224972];
		for (ulong i = 0; i < length; ++i)
			assert ( approxEqual(norm[i], result[i], accuracy) );
	}

