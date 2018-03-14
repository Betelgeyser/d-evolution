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

/**
 * Convenient struct to handle cuBLAS matricies.
 *
 * Row-major order.
 */
struct Matrix
{
	alias values this;
	
	float* values; /// Self explaining.
	uint   rows;   /// ditto
	uint   cols;   /// ditto
	
	/**
	 * Number of elements.
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
	 * Creates matrix and allocates memory on GPU device.
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
	{
		scope(failure) freeMem();
		
		this.rows = rows;
		this.cols = cols;
		
		cudaMallocManaged(values, length);
	}
	
	/// ditto
	this(in uint rows, in uint cols, curandGenerator_t generator) nothrow @nogc
	{
		scope(failure) freeMem();
		
		this(rows, cols);
		curandGenerate(generator, values, length);
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
 * Calculate an Absolute Error between $(D_PARAM A) and $(D_PARAM B) array of vectors on GPU.
 *
 * Resulting array $(D_PARAM error) calculated by formula:
 *
 * <math>
 *   <mrow>
 *     <msub>
 *       <mi>error</mi>
 *       <mi>i</mi>
 *     </msub>
 *     <mo>=</mo>
 *     <msqrt>
 *       <msup>
 *         <mfenced open="(" close=")" separators="">
 *           <msub>
 *             <mi>A</mi>
 *             <mi>i</mi>
 *           </msub>
 *           <mo>-</mo>
 *           <msub>
 *             <mi>B</mi>
 *             <mi>i</mi>
 *           </msub>
 *         </mfenced>
 *         <mn>2</mn>
 *       </msup>
 *     </msqrt>
 *   </mrow>
 * </math>
 *
 * Though $(D_PARAM A) and $(D_PARAM B) are of the type `Matrix` this is a technical convinience. They are interpreted
 * as arrays of vectors where a single col is a single vector.
 *
 * Params:
 *     A = The first array of vectors.
 *     B = The second array of vectors.
 *     error = The resulting array of errors. 
 *     cublasHandle = cuBLAS handle.
 */
void AE(in Matrix A, in Matrix B, ref Matrix error, cublasHandle_t cublasHandle)
in
{
	assert (A.length == B.length);
	assert (A.cols   <= error.cols);
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
	import std.math;
	mixin(writetest!AE);
	
	cublasHandle_t handle;
	cublasCreate(handle);
	scope(exit) cublasDestroy(handle);
	
	auto A = Matrix(3, 4);
	auto B = Matrix(3, 4);
	auto E = Matrix(1, 4);
	
	for (int i = 0; i < A.length; i++)
	{
		A[i] = i;
		B[i] = i * 1.5;
	}
	
	AE(A, B, E, handle);
	cudaDeviceSynchronize();
	
	float[] result = [1.118034, 3.535534, 6.103278, 8.689074];
	for (int i = 0; i < E.length; i++)
		assert ( approxEqual(E[i], result[i], 0.000001) );
}

float MASE(in Matrix measured, in Matrix approximated)
{
//	assert (measured.cols == approximated.cols);
//	assert (measured.rows == approximated.rows);
//	assert (measured.rows >  1);
//	
////	auto naive = Matrix(measured.rows - 1, measured.cols);
////	cudaMemcpy(naive.values, measured.values + 1, measured.rows - 1, cudaMemcpyKind.cudaMemcpyHostToHost);
////	cuda_sub(naive.values, measured.values + 1, measured.rows - 1);
////	
////	auto naive_L2 = Matrix(naive.rows, 1);
////	cuda_L2(naive.values, naive_L2.values, naive.cols, naive.rows);
////	
////	auto error = Matrix(measured.rows, measured.cols);
////	cudaMemcpy(error.values, measured.values, measured.rows, cudaMemcpyKind.cudaMemcpyHostToHost);
////	cuda_sub(naive.values, naive_L2.values, naive.rows);
////	
////	auto error_L2 = cuda_L2(naive.values, naive_L2.values, naive.cols, naive.rows);
////	cuda_L2(naive.values, naive_L2.values, naive.cols, naive.rows);
////	
////	
	return 1;
}

void MAE(in Matrix measured, in Matrix approximated, ref Matrix error)
in
{
	assert (measured.rows == approximated.rows);
	assert (measured.cols == approximated.cols);
}
body
{
	float result = 0;
//	
//	auto error = Matrix(measured.rows, measured.cols);
//	for (int i = 0; i < error.length; i++)
//		error[i] = measured[i];
//	
//	cuda_sub(error.values, measured.values, error.length);
//	
//	auto error_L2 = Matrix(error.rows, 1);
//	cuda_L2(error.values, error_L2.values, error.cols, error.rows);
//	
//	cudaDeviceSynchronize();
//	for (uint i = 0; i < naive_L2.length; i++)
//		result += naive[i] / naive_L2.length;
//	
//	return result;
//}
//
///**
// * Evaluate MAE of naive forecast.
// *
// * Useful for MASE evaluation.
// */
//private float NaiveMAE(in Matrix data)
//{
//	import std.stdio;
//	float result = 0;
//	
//	auto naive = Matrix(data.rows - 1, data.cols);
//	
//	// copy data to naive but one row up
//	for (int i = 0; i < naive.length; i++)
//		naive[i] = data[i + data.cols];
//	
//	cuda_sub(naive.values, data.values, naive.rows);
//	
//	cudaDeviceSynchronize();
//	for (int i = 0; i < naive.length; i++)
//		writeln(naive[i]);
//	
//	auto naive_L2 = Matrix(naive.rows, 1);
//	cuda_L2(naive.values, naive_L2.values, naive.cols, naive.rows);
//	
//	cudaDeviceSynchronize();
//	for (uint i = 0; i < naive_L2.length; i++)
//		result += naive[i] / naive_L2.length;
//	
//	return result;
//}

///
unittest
{
//	import std.math : approxEqual;
//	mixin(writetest!NaiveMAE);
//	
//	/* 0 1 2 *
//	 * 3 4 5 *
//	 * 6 7 8 */
//	auto data = Matrix(3, 3);
//	for (uint i = 0; i < 9; i++)
//		data[i] = i;
//	cudaDeviceSynchronize();
//	
//	writeln(NaiveMAE(data));
//	assert ( approxEqual(NaiveMAE(data), 4.530593, 0.00001) );
}

// TODO: extended matrix. Has additional column filled with 1's which is not affected by activation function.
//struct ExtendedMatrix
//{
//	Matrix mt;
//	alias mt this;
//	
//	this()
//}

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
		import std.math : approxEqual;
		mixin(writetest!cuda_tanh);
		
		float* data;
		cudaMallocManaged(data, 5);
		data[0] = -1_000;
		data[1] =     -1;
		data[2] =      0;
		data[3] =      1;
		data[4] =  1_000;
		
		cuda_tanh(data, 5);
		cudaDeviceSynchronize();
		
		immutable float[] result = [-1.000000, -0.761594, 0.0,  0.761594, 1.000000];
		for (int i = 0; i < 5; i++)
			assert ( approxEqual(data[i], result[i], 0.000001) );
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
		import std.math : approxEqual;
		mixin(writetest!cuda_fill);
		
		float* data;
		cudaMallocManaged(data, 5);
		
		cuda_fill(data,     1, 5);
		cuda_fill(data + 1, 2, 3);
		cudaDeviceSynchronize();
		
		immutable float[] result = [1, 2, 2, 2, 1];
		for (int i = 0; i < 5; i++)
			assert ( approxEqual(data[i], result[i], 0.000001) );
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
		import std.math : approxEqual;
		mixin(writetest!cuda_L2);
		
		float* data;
		float* norm;
		cudaMallocManaged(data, 8);
		cudaMallocManaged(norm, 2);
		
		for (int i = 0; i < 8; i++)
			data[i] = i;
		
		cuda_L2(data, norm, 4, 2);
		cudaDeviceSynchronize();
		
		immutable float[] result = [3.741657, 11.224972];
		for (int i = 0; i < 2; i++)
			assert ( approxEqual(norm[i], result[i], 0.000001) );
	}

//**
// * Vector magnitude in the Euclidean vector space.
// *
// * Params:
// *     vector = Vector.
// */
//double magnitude(in double[] vector) pure nothrow @safe @nogc // Pop, pop!
//{
//	return sqrt(
//		vector.map!"pow(a, 2)".sum
//	);
//}
//
//unittest
//{
//	import std.stdio : writeln;
//	import std.math  : approxEqual;
//	
//	writeln("statistics.magnitude(double[] vector)");
//	
//	assert (approxEqual(
//			magnitude( [100] ),
//			100,
//			0.000_001
//		));
//	
//	assert (approxEqual(
//			magnitude( [0, 0, 0, 0] ),
//			0,
//			0.000_001
//		));
//	
//	assert (approxEqual(
//			magnitude( [4, 3] ),
//			5,
//			0.000_001
//		));
//}
//
///**
// * Absolute error.
// *
// * Params:
// *     vTrue = Actual vector.
// *     vApprox = Approximated vector.
// *
// * Returns:
// *     Absolute error between actual and approximated vectors
// */
//double AE(in double[] vTrue, in double[] vApprox) pure nothrow @safe
//{
//	assert (vTrue.length == vApprox.length);
//	
//	double[] diff;
//	diff.length = vTrue.length;
//	diff[] = vTrue[] - vApprox[];
//	
//	return magnitude(diff);
//}
//
//unittest
//{
//	import std.stdio : writeln;
//	import std.math  : approxEqual;
//	
//	writeln("statistics.AE(in double[] vTrue, in double[] vApprox)");
//	
//	assert (approxEqual(
//			AE([10_000_000.0], [10_000_000.1]),
//			0.1,
//			0.000_001
//		));
//	
//	assert (approxEqual(
//			AE([0.000_000_000_100], [0.000_000_000_101]),
//			0.000_000_000_001_00,
//			0.000_000_000_000_01
//		));
//	
//	assert (approxEqual(
//			AE( [1.0, 1.0], [2.0, 2.0] ),
//			1.414_21,
//			0.000_01
//		));
//}
//
//**
// * Relative error.
// *
// * Params:
// *     vTrue = Actual vector.
// *     vApprox = Approximated vector.
// *
// * Returns:
// *     Relative error between actual and approximated vectors. 
// */
//double RE(in double[] vTrue, in double[] vApprox) pure nothrow @safe
//{
//	return AE(vTrue, vApprox) / magnitude(vTrue);
//}
//
//unittest
//{
//	import std.stdio : writeln;
//	import std.math : approxEqual;
//	
//	writeln("statistics.RE(in double[] vTrue, in double[] vApprox)");
//	
//	assert (approxEqual(
//			RE([10_000_000], [10_000_001]),
//			0.000_000_100,
//			0.000_000_001
//		));
//	
//	assert (approxEqual(
//			RE([0.000_000_000_1], [0.000_000_000_101]),
//			0.01,
//			0.000_001
//		));
//	
//	assert (approxEqual(
//			RE( [3, 4], [3.000_001, 3.999_999] ),
//			0.000_000_283,
//			0.000_000_001
//		));
//}
//
//**
// * Mean absolute relative error.
// *
// * Params:
// *     sTrue = Sample of real data.
// *     sApprox = Sample of approximated data.
// *
// * Returns;
// *     Mean absolute relative error between given data samples.
// */
//double MARE(in double[][] sTrue, in double[][] sApprox) pure nothrow @safe
//in
//{
//	assert (sTrue.length == sApprox.length);
//	foreach (i, v; sTrue)
//		assert (v.length == sApprox[i].length);
//}
//body
//{
//	scope double[] tmp;
//	for (ulong i = 0; i < sTrue.length; i++)
//		tmp ~= RE(sTrue[i], sApprox[i]);
//	
//	return tmp.sum / sTrue.length;
//}
//
//unittest
//{
//	import std.stdio : writeln;
//	import std.math : approxEqual;
//	
//	writeln("statistics.MARE(in double[][] sTrue, in double[][] sApprox)");
//	
//	assert (approxEqual(
//			MARE(
//				[ [1_000_000_000.0], [-2_000_000_000.0], [3_000_000_000.0] ],
//				[ [1_000_000_000.0], [-2_000_000_001.0], [2_999_999_999.0] ]
//			),
//			0.000_000_000_278,
//			0.000_000_000_001
//		));
//	
//	assert (approxEqual(
//			MARE(
//				[ [0.000_000_10], [0.000_000_20], [-0.000_000_30] ],
//				[ [0.000_000_11], [0.000_000_19], [-0.000_000_30] ]
//			),
//			0.050_000,
//			0.000_001
//		));
//}
//
//**
// * Standard error of a given sample.
// *
// * Params:
// *     sample = Data sample.
// */
//double standardError(double[] sample)
//{
//	return pow(sample.map!(x => pow(x - mean(sample), 2)).sum / sample.length, 0.5);
//}
//	
//unittest
//{
//	import std.math : approxEqual;
//	
//	assert (approxEqual(
//			standardError([1_000_000_000, 1_000_000_001, 999_999_999]),
//			0.8165
//		));
//	
//	assert (approxEqual(
//			standardError([0.000_000_000_1, 0.000_000_000_11, 0.000_000_000_09]),
//			0.000_000_000_008_165,
//			0.000_000_000_000_001
//		));
//}
//
