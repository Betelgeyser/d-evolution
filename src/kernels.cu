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

// CUDA Kernel function to compute tanh of each element of an array.
__global__
void kernel_tanh(float *x, size_t n)
{
	for (int i = 0; i < n; i++)
		x[i] = tanhf(x[i]);
}

__host__
void cuda_tanh(float *x, size_t n)
{
	kernel_tanh<<<1, 1>>>(x, n);
}

// CUDA Kernel function to fill array with values.
__global__
void kernel_fill(float *x, float val, size_t count)
{
	for (int i = 0; i < count; i++)
		x[i] = val;
}

__host__
void cuda_fill(float *x, float val, size_t count)
{
	kernel_fill<<<1, 1>>>(x, val, count);
}

// CUDA Kernel subtraction.
__global__
void kernel_sub(float *x, const float *y, size_t n)
{
	for (int i = 0; i < n; i++)
		x[i] = x[i] - y[i];
}

__host__
void cuda_sub(float *x, const float *y, size_t n)
{
	kernel_sub<<<1, 1>>>(x, y, n);
}

// CUDA Kernel L2 norm.
__global__
void kernel_L2(const float *x, float *y, int dim, size_t count)
{
	for (int i = 0; i < count; i++)
		y[i] = normf(dim, x + dim * i);
}

__host__
void cuda_L2(const float *x, float *y, int dim, size_t count)
{
	kernel_L2<<<1, 1>>>(x, y, dim, count);
}

