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

/**
 * Calculate hyperbolic tangent for each element in array on GPU.
 *
 * Params:
 *     x = Array to calculate.
 *     n = Size of array. If n is less than atual x size, only the ferst n elements will be calculated.
 */
__global__
void kernel_tanh(float *x, const size_t n)
{
	for (int i = 0; i < n; ++i)
		x[i] = tanhf(x[i]);
}

/// ditto
__host__
void cuda_tanh(float *x, const size_t n)
{
	kernel_tanh<<<1, 1>>>(x, n);
}

/**
 * Set absolute value for each element in an array in place on GPU.
 *
 * Params:
 *     x = Pointer to an array.
 *     n = Size of array. If n is less than atual x size, only the ferst n elements will be calculated.
 */
__global__
void kernel_abs(float *x, const size_t n)
{
	for (int i = 0; i < n; ++i)
		x[i] = fabsf(x[i]);
}

/// ditto
__host__
void cuda_abs(float *x, const size_t n)
{
	kernel_abs<<<1, 1>>>(x, n);
}

/**
 * Scale [0; 1] to [min; max].
 *
 * Params:
 *     min = Minimum scaled value.
 *     max = Maximum scaled value.
 *     alpha = Value to scale.
 */
__device__
float scale(const float min, const float max, const float alpha)
{
	return alpha * fabsf(max - min) + min;
}

/**
 * Scale array of values from [0; 1] to [min; max].
 *
 * Params:
 *     ptr = Pointer to an array to scale.
 *     min = Minimum scaled value.
 *     max = Maximum scaled value.
 *     count = Number of values to scale.
 */
__global__
void kernel_scale(float *ptr, const float min, const float max, const size_t count)
{
	for (int i = 0; i < count; ++i)
		ptr[i] = scale(min, max, ptr[i]);
}

/// ditto
__host__
void cuda_scale(float *ptr, const float min, const float max, const size_t count)
{
	kernel_scale<<<1, 1>>>(ptr, min, max, count);
}

/**
 * BLX-α crossover.
 *
 * Params:
 *     x, y = Pointers to parent arrays.
 *     offspring = Pointer to an offspring array.
 *     u = Pointer to an array of random uniform values in the range of (0; 1].
 *     alpha = α parameter of BLX-α crossover.
 *     n = Number of values to crossover.
 */
__global__
void kernel_BLX_a(const float *x, const float *y, float *offspring, const float *u, const float alpha, const size_t n)
{
	for (int i = 0; i < n; ++i)
		offspring[i] = scale(
			fminf(x[i], y[i]) - alpha * fabsf(x[i] - y[i]),
			fmaxf(x[i], y[i]) + alpha * fabsf(x[i] - y[i]),
			u[i]
		);
}

/// ditto
__host__
void cuda_BLX_a(const float *x, const float *y, float *offspring, const float *u, const float alpha, const size_t n)
{
	kernel_BLX_a<<<1, 1>>>(x, y, offspring, u, alpha, n);
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
	for (int i = 0; i < count; ++i)
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
void kernel_L2(const float *x, float *y, const int dim, const size_t count)
{
	for (int i = 0; i < count; ++i)
		y[i] = normf(dim, x + dim * i);
}

/// ditto
__host__
void cuda_L2(const float *x, float *y, const int dim, const size_t count)
{
	kernel_L2<<<1, 1>>>(x, y, dim, count);
}

