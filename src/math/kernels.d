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
 *
 * This module contains basic structs, subroutines and cuda kernel interfaces for mathematics.
 */
module math.kernels;

// CUDA modules
import cuda.cudaruntimeapi;

// DNN modules
import common;
import math.matrix;

version (unittest)
{
	import std.algorithm : each, equal;
	import std.math      : approxEqual;
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
private extern (C++) void cuda_tanh(float* x, const size_t n) nothrow @nogc;
void cudaTanh(float[] x) nothrow @nogc
{
	cuda_tanh(x.ptr, x.length);
}

///
unittest
{
	mixin(writetest!cudaTanh);
	
	immutable length = 5;
	
	float[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	data[0 .. $] = [-1_000, -1, 0, 1, 1_000];
	
	cudaTanh(data);
	cudaDeviceSynchronize();
	
	immutable float[] result = [-1.000000, -0.761594, 0.000000,  0.761594, 1.000000];
	assert (equal!approxEqual(data, result));
}

/**
 * Set absolute value for each element in an array in place on GPU.
 *
 * Params:
 *     x = Pointer to an array.
 *     n = Size of array. If n is less than atual x size, only the ferst n elements will be calculated.
 */
private extern (C++) void cuda_scale(void* ptr, const float  a, const float  b, const size_t count) nothrow @nogc;
const(float[]) cudaScale(uint[] x, in float a, in float b) nothrow @nogc
in
{
	assert (a <= b);
}
body
{
	cuda_scale(x.ptr, a, b, x.length);
	return cast(const(float)[])x;
}

///
unittest
{
	mixin(writetest!cudaScale);
	
	immutable length = 3;
	
	uint[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	data[0 .. $]   = [uint.min, uint.max / 2, uint.max];
	float[] result = [      -1,            0,        1];
	
	const(float)[] fData = cudaScale(data, -1, 1); // fData is just a copy pointer
	cudaDeviceSynchronize();
	
	assert(equal!approxEqual(fData, result));
	
	data[0 .. $] = [uint.min, uint.max / 2,  uint.max];
	result       = [       0,      500_000, 1_000_000];
	
	fData = cudaScale(data, 0, 1_000_000);
	cudaDeviceSynchronize();
	
	assert(equal!approxEqual(fData, result));
}

/**
 * BLX-α crossover.
 *
 * Params:
 *     x = Pointer to a parent array.
 *     y = Pointer to a parent array.
 *     offspring = Pointer to an offspring array.
 *     alpha = α parameter of BLX-α crossover.
 *     u = Pointer to an array of random uniform values in the range of [0; 1].
 *     n = Number of values to crossover.
 */
void cuda_BLX_a(const(float*) x, const(float*) y, float* offspring, const float alpha, const(float*) u, const size_t n) nothrow @nogc;

///
unittest
{
	mixin(writetest!cuda_BLX_a);
	
	import std.math : approxEqual;
	
	immutable accuracy = 0.000_001;
	immutable length   = 3;
	immutable alpha    = 0.2;
	
	// Initialize parents
	float* x;
	cudaMallocManaged(x, length);
	scope(exit) cudaFree(x);
	
	float* y;
	cudaMallocManaged(y, length);
	scope(exit) cudaFree(y);
	
	x[0..length] = [-1, 0, 1];
	y[0..length] = [ 2, 0, 0];
	
	// An offspring does not need to be initialized, just allocate memory
	float* offspring;
	cudaMallocManaged(offspring, length);
	scope(exit) cudaFree(offspring);
	
	// There should be pregenerated random values in the range [0; 1]
	float* u;
	cudaMallocManaged(u, length);
	scope(exit) cudaFree(u);
	u[0..length] = [0.0, 0.2, 0.8];
	
	// Artificial crossover. It will be more random in the wilderness
	cuda_BLX_a(x, y, offspring, alpha, u, length);
	cudaDeviceSynchronize();
	
	immutable float[] result = [-1.6, 0, 0.92];
	for (ulong i = 0; i < length; ++i)
		assert ( approxEqual(offspring[i], result[i], accuracy) );
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
private extern(C++) void cuda_fill(float* x, const float val, const size_t n) nothrow @nogc;
void cudaFill(float[] x, in float val) nothrow @nogc
{
	cuda_fill(x.ptr, val, x.length);
}

///
unittest
{
	mixin(writetest!cudaFill);
	
	immutable length = 5;
	
	float[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	cudaFill(data,           1);
	cudaFill(data[1 .. $-1], 2);
	cudaDeviceSynchronize();
	
	immutable float[] result = [1, 2, 2, 2, 1];
	assert (equal!approxEqual(data, result));
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
private extern(C++) void cuda_L2(const(float)* x, float* y, const uint dim, const size_t count) nothrow @nogc;
void cudaL2(const Matrix x, float[] y) nothrow @nogc
in
{
	assert (x.cols == y.length);
}
body
{
	cuda_L2(x.ptr, y.ptr, x.rows, x.cols);
}

///
unittest
{
	mixin(writetest!cudaL2);
	
	immutable dim    = 4;
	immutable length = 2;
	
	Matrix data = Matrix(dim, length);
	scope(exit) data.freeMem();
	
	float[] norm;
	cudaMallocManaged(norm, length);
	scope(exit) cudaFree(norm);
	
	data.values.each!"a = i";
	
	cudaL2(data, norm);
	cudaDeviceSynchronize();
	
	immutable float[] result = [3.741657, 11.224972];
	assert (equal!approxEqual(norm, result));
}

