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
 * This file contains implementations and interfaces for cuda kernels. They are compiled to a standalone dynamic library
 * to be linked with D code later.
 */


const float uint_max_fp = 4294967295.0f; /// Maximum value of unsigned integer represented in floating point format.

/**
 * Calculate hyperbolic tangent for each element in an array in place on GPU.
 *
 * Params:
 *     x = Pointer to an array.
 *     n = Size of array. If n is less than atual x size, only the ferst n elements will be calculated.
 */
__global__
void kernel_tanh(float *x, const size_t n)
{
	for (size_t i = 0; i < n; ++i)
		x[i] = tanhf(x[i]);
}

/// ditto
__host__
void cuda_tanh(float *x, const size_t n)
{
	kernel_tanh<<<1, 1>>>(x, n);
}

/**
 * Returns a floating point value scaled from unsigned integer number x to a given segment [a; b]
 * 
 * Params:
 *     x = Value to scale.
 *     a = Left bound.
 *     b = Right bound.
 */
__device__
float scale(const unsigned int x, const float a, const float b)
{
	return a + (b - a) * (float)x / uint_max_fp;
}

/**
 * Transform uniformly distrubuted random bits into uniformly distributed
 * random floating point number in range [a; b], where a <= b.
 * The transformation is performed in place.
 *
 * Params:
 *     x = Pointer to an array.
 *     n = Size of array. If n is less than atual x size, only the ferst n elements will be calculated.
 */
__global__
void kernel_scale(void *ptr, const float a, const float b, const size_t count)
{
	unsigned int *uPtr = (unsigned int*)ptr;
	float        *fPtr = (float*)ptr;
	
	for (size_t i = 0; i < count; ++i)
		fPtr[i] = scale(uPtr[i], a, b);
}

/// ditto
__host__
void cuda_scale(void *ptr, const float a, const float b, const size_t count)
{
	kernel_scale<<<1, 1>>>(ptr, a, b, count);
}

/**
 * BLX-α crossover.
 *
 * Params:
 *     x, y = Pointers to parent arrays.
 *     offspring = Pointer to an offspring array.
 *     alpha = α parameter of BLX-α crossover.
 *     u = Pointer to an array of random uniform values in the range of [0; 1].
 *     n = Number of values to crossover.
 */
__global__
void kernel_BLX_a(
	const float *x, const float *y,
	float *offspring,
	const float a, const float b,
	const float alpha,
	const unsigned int *u,
	const size_t n
)
{
	for (size_t i = 0; i < n; ++i)
	{
		float _a = fminf(x[i], y[i]) - alpha * fabsf(x[i] - y[i]);
		float _b = fmaxf(x[i], y[i]) + alpha * fabsf(x[i] - y[i]);
		
		offspring[i] = scale(u[i], _a, _b);
		
		if (offspring[i] < a)
			offspring[i] = a;
		
		if (offspring[i] > b)
			offspring[i] = b;
	}
}

/// ditto
__host__
void cuda_BLX_a(
	const float *x, const float *y,
	float *offspring,
	const float a, const float b,
	const float alpha,
	const unsigned int *u,
	const size_t n
)
{
	kernel_BLX_a<<<1, 1>>>(x, y, offspring, a, b, alpha, u, n);
}

/**
 * Fill array on GPU.
 *
 * Params:
 *     x = A pointer to an array. Could point to not the first element.
 *     val = A value to fill with.
 *     n = Size of the array. If n is less than the actual x size, only the first n elements starting from the pointer p
 *         will be filled.
 */
__global__
void kernel_fill(float *x, const float val, const size_t count)
{
	for (size_t i = 0; i < count; ++i)
		x[i] = val;
}

/// ditto
__host__
void cuda_fill(float *x, const float val, const size_t count)
{
	kernel_fill<<<1, 1>>>(x, val, count);
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
__global__
void kernel_L2(const float *x, float *y, const unsigned int dim, const size_t count)
{
	for (size_t i = 0; i < count; ++i)
		y[i] = normf(dim, x + dim * i);
}

/// ditto
__host__
void cuda_L2(const float *x, float *y, const unsigned int dim, const size_t count)
{
	kernel_L2<<<1, 1>>>(x, y, dim, count);
}

