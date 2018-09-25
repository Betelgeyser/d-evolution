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

// Standard D modules
import std.algorithm   : mean;
import std.parallelism : taskPool;
import std.string      : format;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;

// DNN modules
import common;
import math.matrix;
import math.kernels;

version (unittest)
{
	import std.algorithm : each, equal;
	import std.math      : approxEqual;
	
	private cublasHandle_t cublasHandle;
	
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
void AE(in Matrix A, in Matrix B, Matrix error, cublasHandle_t cublasHandle)
{
	if (A.rows != B.rows || A.cols != B.cols)
		throw new Error("Input matricies must have same size, got %dx%d and %dx%d.".format(A.rows, A.cols, B.rows, B.cols));
	
	if (A.cols != error.cols || error.rows != 1)
		throw new Error("Output matrix must be 1x%d, got %dx%d".format(A.cols, error.rows, error.cols));
	
	immutable float alpha =  1;
	immutable float beta  = -1;
	
	auto C = Matrix(A.rows, A.cols);
	scope(exit) C.freeMem();
	
	geam(1, A, -1, B, C, cublasHandle);
	cudaL2(C, error);
}

///
unittest
{
	mixin(writeTest!AE);
	
	immutable cols = 3;
	immutable rows = 4;
	
	auto A = Matrix(cols, rows);
	auto B = Matrix(cols, rows);
	auto E = Matrix(1,    rows);
	
	A.each!"a = i";
	B.each!"a = 1.5 * i";
	
	AE(A, B, E, cublasHandle);
	cudaDeviceSynchronize();
	
	immutable float[] result = [1.118034, 3.535534, 6.103278, 8.689074];
	assert (equal!approxEqual(E.values, result));
}

/**
 * Calculate a Mean Absolute Error between $(D_PARAM A) and $(D_PARAM B) array of vectors on GPU.
 *
 * Though $(D_PARAM A) and $(D_PARAM B) are of the type `Matrix` this is a technical convinience. They are interpreted
 * as arrays of vectors where a single column is a single vector.
 *
 * Calls cudaDeviceSyncronize() internally.
 *
 * Params:
 *     A = The first array of vectors.
 *     B = The second array of vectors.
 *     cublasHandle = cuBLAS handle.
 *
 * See_also:
 *     $(LINK https://en.wikipedia.org/wiki/Mean_absolute_error)
 */
float MAE(in Matrix A, in Matrix B, cublasHandle_t cublasHandle)
{
	if (A.rows != B.rows || A.cols != B.cols)
		throw new Error("Input matricies must have same size, got %dx%d and %dx%d.".format(A.rows, A.cols, B.rows, B.cols));
	
	auto error = Matrix(1, A.cols);
	scope(exit) error.freeMem();
	
	AE(A, B, error, cublasHandle);
	cudaDeviceSynchronize();
	
	// TODO: less accurate than std.algorith.mean but much faster.
	// Though, multithreaded mean shoud be used.
	return taskPool.reduce!"a + b"(error.values) / error.values.length;
}

///
unittest
{
	mixin(writeTest!MAE);
	
	immutable rows = 3;
	immutable cols = 4;
	
	auto A = Matrix(rows, cols);
	scope(exit) A.freeMem();
	
	auto B = Matrix(rows, cols);
	scope(exit) B.freeMem();
	
	A.each!"a = i";
	B.each!"a = 1.5 * i";
	
	float error = MAE(A, B, cublasHandle);
	
	assert (approxEqual(error, 4.861480));
}

/**
 * TODO: docs, GPU, unittest
 */
float MPE(in Matrix A, in Matrix B, cublasHandle_t cublasHandle)
in
{
	assert (A.cols == B.cols);
	assert (A.rows == B.rows);
}
body
{
	auto error = Matrix(1, A.cols);
	scope(exit) error.freeMem();
	
	auto C = Matrix(1, A.cols);
	scope(exit) C.freeMem();
	
	AE(A, B, error, cublasHandle);
	cudaL2(A, C);
	cudaDeviceSynchronize();
	
	float result = 0;
	foreach (i, v; error.values)
		result += error[i] / C[i] / error.length;
	
	return result * 100;
}

/**
 * Calculate a Mean Absolute Error of naive forecast on GPU.
 *
 * Useful for MASE calculation.
 *
 * Though $(D_PARAM data) is of the type `Matrix` this is a technical convinience. It is interpreted as an array of vectors
 * where a single column is a single vector.
 *
 * Calls cudaDeviceSyncronize() internally.
 *
 * Params:
 *     data = An array of input vectors.
 *     cublasHandle = cuBLAS handle.
 */
float MAENaive(in Matrix data, cublasHandle_t cublasHandle)
{
	auto measured = data.colSlice(0, data.cols - 1);
	auto naive    = data.colSlice(1, data.cols);
	
	return MAE(measured, naive, cublasHandle);
}

///
unittest
{
	mixin(writeTest!MAENaive);
	
	auto data = Matrix(2, 3);
	scope(exit) data.freeMem();
	
	data.each!"a = i * i";
	
	float error = MAENaive(data, cublasHandle);
	
	assert (approxEqual(error, 14.472136));
}

/**
 * Calculate a Mean Absolute Scalde Error between $(D_PARAM measured) and $(D_PARAM approximated) arrays of vectors on GPU.
 *
 * Though data is of the type `Matrix` this is a technical convinience. It is interpreted as an array of vectors
 * where a single column is a single vector.
 *
 * Calls cudaDeviceSyncronize() internally.
 *
 * Params:
 *     measured = An array of vectors of measured/actual/real data.
 *     approximated = An array of vectors of approximated/estimated data.
 *     cublasHandle = cuBLAS handle.
 */
float MASE(in Matrix measured, in Matrix approximated, cublasHandle_t cublasHandle)
{
	if (measured.rows != approximated.rows || measured.cols != approximated.cols)
		throw new Error(
			"Input matricies must have same size, got %dx%d and %dx%d."
			.format(measured.rows, measured.cols, approximated.rows, approximated.cols)
		);
	
	return MAE(measured, approximated, cublasHandle) / MAENaive(measured, cublasHandle);
}

///
unittest
{
	mixin(writeTest!MASE);
	
	immutable cols = 3;
	immutable rows = 4;
	
	auto measured = Matrix(cols, rows);
	scope(exit) measured.freeMem();
	
	auto approximated = Matrix(cols, rows);
	scope(exit) approximated.freeMem();
	
	measured.each!"a = i";
	approximated.each!"a = i + 1";
	
	float error = MASE(measured, approximated, cublasHandle);
	
	assert (approxEqual(error, 0.333333));
}

