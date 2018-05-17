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


/**
 * Calculate hyperbolic tangent for each element of an array `x` on a GPU in place.
 *
 * Params:
 *     x = An array to calculate.
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
 * Transform uniformly distrubuted random bits into uniformly distributed random floating point numbers in range [a; b],
 * where a <= b.
 *
 * Due to implementation details it is not recomended to pass a and b close to ±1.0e28.
 *
 * Params:
 *     x = Array of random bits.
 *     a = Minimum value.
 *     b = Maximum value.
 *
 * Returns:
 *     a new pointer to the array `x` of float valus.
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
private extern (C++) void cuda_BLX_a(
	const(float*) x, const(float*) y,
	float* offspring,
	const float a, const float b,
	const float alpha,
	const(uint*) u,
	const size_t n
) nothrow @nogc;
void cudaBLXa(in float[] x, in float[] y, float[] offspring, in float a, in float b, const float alpha, in uint[] u) nothrow @nogc
in
{
	assert (offspring.length == x.length);
	assert (offspring.length == y.length);
	assert (offspring.length == u.length);
	
	assert (alpha >= 0 && alpha <= 1);
}
body
{
	cuda_BLX_a(x.ptr, y.ptr, offspring.ptr, a, b, alpha, u.ptr, offspring.length);
}

///
unittest
{
	mixin(writetest!cudaBLXa);
	
	immutable length = 3;
	immutable alpha  = 0.5;
	
	// Initialize parents
	float[] x;
	cudaMallocManaged(x, length);
	scope(exit) cudaFree(x);
	
	float[] y;
	cudaMallocManaged(y, length);
	scope(exit) cudaFree(y);
	
	x[0 .. $] = [0, 1, -1];
	y[0 .. $] = [0, 0,  2];
	
	// An offspring does not need to be initialized, just allocate memory
	float[] offspring;
	cudaMallocManaged(offspring, length);
	scope(exit) cudaFree(offspring);
	
	// There should be pregenerated random values in the range [0; 1]
	uint[] u;
	cudaMallocManaged(u, length);
	scope(exit) cudaFree(u);
	
	u[0 .. $] = [0, uint.max / 2, uint.max];
	
	// Artificial crossover. u will be random in real calcilations.
	cudaBLXa(x, y, offspring, -10, 10, alpha, u);
	cudaDeviceSynchronize();
	
	float[] result = [0.0, 0.5, 3.5];
	assert (equal!approxEqual(offspring, result));
	
	// Clamp test
	cudaBLXa(x, y, offspring, -2, 2, alpha, u);
	cudaDeviceSynchronize();
	
	result = [0.0, 0.5, 2.0];
	assert (equal!approxEqual(offspring, result));
}

/**
 * Fill an array on a GPU with `val`.
 *
 * Params:
 *     x = An array to fill.
 *     val = A value to fill with.
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
 * Per-vector calculation of the Euclidean distance (L2 norm) of a vector array on a GPU.
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

