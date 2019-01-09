/**
 * Copyright © 2017 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
 *     actual = Matrix of actual values.
 *     predicted = Matrix of predicted values.
 *     error = The resulting array of errors. 
 *     cublasHandle = cuBLAS handle.
 */
void AE(in Matrix actual, in Matrix predicted, Matrix error, cublasHandle_t cublasHandle)
{
	if (actual.rows != predicted.rows || actual.cols != predicted.cols)
		throw new Error(
			"Input matrices have diffent sizes %dx%d and %dx%d"
			.format(actual.rows, actual.cols, predicted.rows, predicted.cols)
		);
	
	if (actual.rows != error.rows)
		throw new Error("Input and output matrices have different number of rows %d and %d.".format(actual.rows, error.rows));
	
	if (error.cols != 1)
		throw new Error("Output matrix must be only 1 column width, not %d.".format(error.cols));
	
	immutable float alpha =  1;
	immutable float beta  = -1;
	
	auto absError = Matrix(actual.rows, actual.cols);
	scope(exit) absError.freeMem();
	
	geam(1, actual, false, -1, predicted, false, absError, cublasHandle);
	cudaL2(absError, error);
}

///
unittest
{
	mixin(writeTest!AE);
	
	immutable rows = 4;
	immutable cols = 3;
	
	auto A = Matrix(rows, cols);
	scope(exit) A.freeMem();
	
	auto P = Matrix(rows, cols);
	scope(exit) P.freeMem();
	
	auto E = Matrix(rows, 1);
	scope(exit) E.freeMem();
	
	A.each!"a = i";
	P.each!"a = 1.5 * i";
	
	AE(A, P, E, cublasHandle);
	cudaDeviceSynchronize();
	
	immutable float[] result = [4.472136, 5.172040, 5.916080, 6.689544];
	assert (equal!approxEqual(E.values, result));
}

/**
 * Calculate the Mean Absolute Error between $(D_PARAM A) and $(D_PARAM B) arrays of vectors on GPU.
 *
 * Though $(D_PARAM A) and $(D_PARAM B) are of the type `Matrix` this is a technical convinience. They are interpreted
 * as arrays of vectors where a single column is a single vector.
 *
 * Calls cudaDeviceSyncronize() internally.
 *
 * Params:
 *     actual = Matrix of actual values.
 *     predicted = Matrix of predicted values.
 *     cublasHandle = cuBLAS handle.
 *
 * See_also:
 *     $(LINK https://en.wikipedia.org/wiki/Mean_absolute_error)
 */
float MAE(in Matrix actual, in Matrix predicted, cublasHandle_t cublasHandle)
{
	if (actual.rows != predicted.rows || actual.cols != predicted.cols)
		throw new Error(
			"Input matrices have different sizes %dx%d and %dx%d."
			.format(actual.rows, actual.cols, predicted.rows, predicted.cols)
		);
	
	auto error = Matrix(actual.rows, 1);
	scope(exit) error.freeMem();
	
	AE(actual, predicted, error, cublasHandle);
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
	
	auto P = Matrix(rows, cols);
	scope(exit) P.freeMem();
	
	A.each!"a = i";
	P.each!"a = 1.5 * i";
	
	float error = MAE(A, P, cublasHandle);
	
	assert (approxEqual(error, 6.456302));
}

/**
 * Calculate the Mean Percentage Error between $(D_PARAM A) and $(D_PARAM B) arrays of vectors on GPU.
 *
 * See_also:
 *     $(LINK https://en.wikipedia.org/wiki/Mean_percentage_error)
 */
float MPE(in Matrix A, in Matrix B, cublasHandle_t cublasHandle)
{
	if (A.rows != B.rows || A.cols != B.cols)
		throw new Error("Input matricies must have same size, got %dx%d and %dx%d.".format(A.rows, A.cols, B.rows, B.cols));
	
	auto error = Matrix(A.rows, 1);
	scope(exit) error.freeMem();
	
	auto C = Matrix(A.rows, 1);
	scope(exit) C.freeMem();
	
	AE(A, B, error, cublasHandle);
	cudaL2(A, C);
	cudaDeviceSynchronize();
	
	float result = 0;
	foreach (i, v; error.values)
		result += error[i] / C[i] / error.length;
	
	return result * 100;
}

// /**
// * Calculate a Mean Absolute Error of naive forecast on GPU.
// *
// * Useful for MASE calculation.
// *
// * Though $(D_PARAM data) is of the type `Matrix` this is a technical convinience. It is interpreted as an array of vectors
// * where a single column is a single vector.
// *
// * Calls cudaDeviceSyncronize() internally.
// *
// * Params:
// *     data = An array of input vectors.
// *     cublasHandle = cuBLAS handle.
// */
//float MAENaive(in Matrix data, cublasHandle_t cublasHandle)
//{
//	auto measured = data.colSlice(0, data.cols - 1);
//	auto naive    = data.colSlice(1, data.cols);
//	
//	return MAE(measured, naive, cublasHandle);
//}
//
// ///
//unittest
//{
//	mixin(writeTest!MAENaive);
//	
//	auto data = Matrix(2, 3);
//	scope(exit) data.freeMem();
//	
//	data.each!"a = i * i";
//	
//	float error = MAENaive(data, cublasHandle);
//	writeln(error);
//	
//	assert (approxEqual(error, 10.344080));
//}
//
// /**
// * Calculate a Mean Absolute Scalde Error between $(D_PARAM measured) and $(D_PARAM approximated) arrays of vectors on GPU.
// *
// * Though data is of the type `Matrix` this is a technical convinience. It is interpreted as an array of vectors
// * where a single column is a single vector.
// *
// * Calls cudaDeviceSyncronize() internally.
// *
// * Params:
// *     measured = An array of vectors of measured/actual/real data.
// *     approximated = An array of vectors of approximated/estimated data.
// *     cublasHandle = cuBLAS handle.
// */
//float MASE(in Matrix measured, in Matrix forecasted, cublasHandle_t cublasHandle)
//{
//	if (measured.rows != forecasted.rows || measured.cols != forecasted.cols)
//		throw new Error(
//			"Input matricies must have same size, got %dx%d and %dx%d."
//			.format(measured.rows, measured.cols, forecasted.rows, forecasted.cols)
//		);
//	
//	return MAE(measured, forecasted, cublasHandle) / MAENaive(measured, cublasHandle);
//}
//
// ///
//unittest
//{
//	mixin(writeTest!MASE);
//	
//	immutable rows = 4;
//	immutable cols = 3;
//	
//	auto measured = Matrix(rows, cols);
//	scope(exit) measured.freeMem();
//	
//	auto forecasted = Matrix(rows, cols);
//	scope(exit) forecasted.freeMem();
//	
//	measured.each!"a = i * i";
//	forecasted.each!"a = i";
//	
//	float error = MASE(measured, forecasted, cublasHandle);
//	
//	assert (approxEqual(error, 0.583906));
//}

