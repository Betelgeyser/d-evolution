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
module math.statistics;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common;
import math.matrix;
import math.kernels;

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
void AE(in Matrix A, in Matrix B, Matrix error, cublasHandle_t cublasHandle) nothrow @nogc
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
	
	geam(1, A, -1, B, C, cublasHandle);
	cuda_L2(C, error, C.rows, error.cols);
}

///
unittest
{
	mixin(writetest!AE);
	
	
	auto A = Matrix(3, 4);
	auto B = Matrix(3, 4);
	auto E = Matrix(1, 4);
	
	for (ulong i = 0; i < A.length; ++i)
	{
		A[i] = i;
		B[i] = i * 1.5;
	}
	
	AE(A, B, E, cublasHandle);
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
float MAENaive(in Matrix data, cublasHandle_t cublasHandle) nothrow @nogc
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
	mixin(writetest!MAENaive);
	
	import std.math : approxEqual;
	immutable accuracy = 0.000_001;
	
	
	auto data = Matrix(2, 3);
	for (ulong i = 0; i < data.length; ++i)
		data[i] = i * i;
	
	assert ( approxEqual(MAENaive(data, handle), 14.472136, accuracy) );
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
	return MAE(measured, approximated, cublasHandle) / MAENaive(measured, cublasHandle);
}

///
unittest
{
	mixin(writetest!MASE);
	
	
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

