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

/**
 * Calculate hyperbolic tangent for each element in array on GPU.
 *
 * Params:
 *     x = Array to calculate.
 *     n = Size of array. If n is less than atual x size, only the ferst n elements will be calculated.
 */
__global__
void kernel_tanh(float *x, size_t n)
{
	for (int i = 0; i < n; i++)
		x[i] = tanhf(x[i]);
}

/// ditto
__host__
void cuda_tanh(float *x, size_t n)
{
	kernel_tanh<<<1, 1>>>(x, n);
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
void kernel_fill(float *x, float val, size_t count)
{
	for (int i = 0; i < count; i++)
		x[i] = val;
}

/// ditto
__host__
void cuda_fill(float *x, float val, size_t count)
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
void kernel_L2(const float *x, float *y, int dim, size_t count)
{
	for (int i = 0; i < count; i++)
		y[i] = normf(dim, x + dim * i);
}

/// ditto
__host__
void cuda_L2(const float *x, float *y, int dim, size_t count)
{
	kernel_L2<<<1, 1>>>(x, y, dim, count);
}

