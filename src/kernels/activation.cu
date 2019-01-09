/**
 * Copyright © 2018 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
 * This file contains implementations and interfaces for cuda kernel activation functions. All cuda kernels then are
 * compiled to a standalone dynamic library to be linked with D code later.
 *
 * Authors: Sergei Iurevich Filippov. $(COPYRIGHT)
 */


/**
 * Calculate the hyperbolic tangent of each element of an array x on a GPU in place.
 *
 * <math><mrow>
 *     <mi mathvariant="italic">tanh</mi><mfenced><mi>x</mi></mfenced>
 *     <mo>=</mo>
 *     <mfrac>
 *         <mrow><msup><mi>e</mi><mrow><mn>2</mn><mi>x</mi></mrow></msup><mo>-</mo><mn>1</mn></mrow>
 *         <mrow><msup><mi>e</mi><mrow><mn>2</mn><mi>x</mi></mrow></msup><mo>+</mo><mn>1</mn></mrow>
 *     </mfrac>
 * </mrow></math>
 *
 * Params:
 *     x = A pointer to an array to calculate.
 *     count = Size of the array.
 */
__global__
void kernel_tanh(float *x, const size_t count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
		x[i] = tanhf(x[i]);
}

/// ditto
__host__
void cuda_tanh(float *x, const size_t count)
{
	kernel_tanh<<<(count + 1023) / 1023, 1024>>>(x, count);
}

/**
 * Calculate the rectified linear unit of each element of an array x on a GPU in place.
 *
 * <math><mrow>
 *     <mi mathvariant="italic">ReLU</mi><mfenced><mi>x</mi></mfenced>
 *     <mo>=</mo>
 *     <mo>{</mo>
 *         <mtable>
 *             <mtr>
 *                 <mtd><mn>0</mn><mi>x</mi></mtd><mtd><mtext>for&nbsp;</mtext><mi>x</mi><mo>&lt;</mo><mn>0</mn></mtd>
 *             </mtr>
 *             <mtr>
 *                 <mtd><mi>x</mi></mtd><mtd><mtext>for&nbsp;</mtext><mi>x</mi><mo>&ge;</mo><mn>0</mn></mtd>
 *             </mtr>
 *         </mtr>
 * </mrow></math>
 *
 * Params:
 *     x = A pointer to an array to calculate.
 *     count = Size of the array.
 */
__global__
void kernel_ReLU(float *x, const size_t count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
		x[i] = fmaxf(0.0f, x[i]);
}

/// ditto
__host__
void cuda_ReLU(float *x, const size_t count)
{
	kernel_ReLU<<<(count + 1023) / 1023, 1024>>>(x, count);
}

/**
 * Calculate the leaky rectified linear unit of each element of an array x on a GPU in place.
 *
 * <math><mrow>
 *     <mi mathvariant="italic">Leaky ReLU</mi><mfenced><mi>x</mi></mfenced>
 *     <mo>=</mo>
 *     <mo>{</mo>
 *         <mtable>
 *             <mtr>
 *                 <mtd><mn>0.01</mn><mi>x</mi></mtd><mtd><mtext>for&nbsp;</mtext><mi>x</mi><mo>&lt;</mo><mn>0</mn></mtd>
 *             </mtr>
 *             <mtr>
 *                 <mtd><mi>x</mi></mtd><mtd><mtext>for&nbsp;</mtext><mi>x</mi><mo>&ge;</mo><mn>0</mn></mtd>
 *             </mtr>
 *         </mtr>
 * </mrow></math>
 *
 * Params:
 *     x = A pointer to an array to calculate.
 *     count = Size of the array.
 */
__global__
void kernel_LeakyReLU(float *x, const size_t count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
		if (x[i] < 0)
			x[i] *= 0.01f;
}

/// ditto
__host__
void cuda_LeakyReLU(float *x, const size_t count)
{
	kernel_LeakyReLU<<<(count + 1023) / 1023, 1024>>>(x, count);
}


