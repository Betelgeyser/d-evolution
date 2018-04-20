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
import std.algorithm : each, mean;

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
	cuda_L2(C.ptr, error.ptr, C.rows, error.cols);
}

///
unittest
{
	mixin(writetest!AE);
	
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
	foreach (i, e; E)
		assert (approxEqual(e, result[i], accuracy));
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
	auto error = Matrix(1, A.cols);
	
	AE(A, B, error, cublasHandle);
	cudaDeviceSynchronize();
	
	return error.values.mean;
}

///
unittest
{
	mixin(writetest!MAE);
	
	immutable rows = 3;
	immutable cols = 4;
	
	auto A = Matrix(rows, cols);
	scope(exit) A.freeMem();
	
	auto B = Matrix(rows, cols);
	scope(exit) B.freeMem();
	
	A.each!"a = i";
	B.each!"a = 1.5 * i";
	
	assert ( approxEqual(MAE(A, B, cublasHandle), 4.861480, accuracy) );
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
	auto measured = data.colSlice(0, data.cols - 1);
	auto naive    = data.colSlice(1, data.cols);
	
	return MAE(measured, naive, cublasHandle);
}

///
unittest
{
	mixin(writetest!MAENaive);
	
	auto data = Matrix(2, 3);
	scope(exit) data.freeMem();
	
	data.each!"a = i * i";
	
	assert ( approxEqual(MAENaive(data, cublasHandle), 14.472136, accuracy) );
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
	
	immutable cols = 3;
	immutable rows = 4;
	
	auto measured = Matrix(cols, rows);
	scope(exit) measured.freeMem();
	
	auto approximated = Matrix(cols, rows);
	scope(exit) approximated.freeMem();
	
	measured.each!"a = i";
	approximated.each!"a = i + 1";
	
	assert ( approxEqual(MASE(measured, approximated, cublasHandle), 0.333333, accuracy) );
}

