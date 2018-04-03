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
 */
module math.kernels;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;

// DNN modules
import common;


extern (C++):

	/**
 * Calculate hyperbolic tangent for each element of an array `x` on GPU.
 *
 * Params:
 *     x = A pointer to an array. Could point to not the first element.
 *     n = Size of the array. If n is less than the actual x size, only the first n elements starting from the pointer p
 *         will be calculated.
 */
void cuda_tanh(float* x, in size_t n) nothrow @nogc;

///
unittest
{
	mixin(writetest!cuda_tanh);
	
	import std.math : approxEqual;
	
	immutable accuracy = 0.000_001;
	immutable length   = 5;
	
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
void cuda_fill(float* x, in float val, in size_t n) nothrow @nogc;

///
unittest
{
	mixin(writetest!cuda_fill);
	
	import std.math : approxEqual;
	
	immutable accuracy = 0.000_001;
	immutable length   = 5;
	
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
void cuda_L2(const(float)* x, float* y, in int dim, in size_t count) nothrow @nogc;

///
unittest
{
	mixin(writetest!cuda_L2);
	
	import std.math : approxEqual;
	
	immutable accuracy = 0.000_001;
	immutable dim      = 4;
	immutable length   = 2;
	
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

/**
 * Scale (0; 1] to (min; max].
 *
 * Params:
 *     ptr = Pointer to an array to scale.
 *     min = Minimum scaled value.
 *     max = Maximum scaled value.
 *     count = Number of values to scale.
 */
void cuda_scale(float* ptr, in float min, in float max, in size_t count);

///
unittest
{
	mixin(writetest!cuda_scale);
	
	import std.math : approxEqual;
	
	immutable accuracy = 0.000_001;
	immutable min      = -200;
	immutable max      =  600;
	immutable length   =    5;
	
	float* data;
	cudaMallocManaged(data, length);
	
	for (ulong i = 0; i < length; ++i)
		data[i] = cast(float) i / length;
	
	cuda_scale(data, min, max, length);
	cudaDeviceSynchronize();
	
	immutable float[] result = [-200, -40, 120, 280, 440];
	for (ulong i = 0; i < length; ++i)
		assert ( approxEqual(data[i], result[i], accuracy) );
}

