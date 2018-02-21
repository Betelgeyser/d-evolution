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

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void kernel_tanh(float *x, int n)
{
	for (int i = 0; i < n; i++)
		x[i] = tanhf(x[i]);
}

__host__
void cuda_tanh(float *x, int n)
{
	kernel_tanh<<<1, 1>>>(x, n);
}

