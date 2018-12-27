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
 * Calculate hyperbolic tangent of each element of an array x on a GPU in place.
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
 * Calculate rectifier of each element of an array x on a GPU in place.
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
 * Calculate leaky rectifier of each element of an array x on a GPU in place.
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

/**
 * Returns a floating point value scaled from unsigned integer number x to a given segment [a; b],
 * meaning 0 will return a and MAX(unsigned int) will return b.
 * 
 * Due to implementation details it is not recomended to pass a and b close to ±1.0e28 as that will
 * cause function to return infinity.
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
 * Transform uniformly distrubuted random bits into uniformly distributed random floating point numbers in range 
 * [a; b], where a <= b. 0 will translate to a and MAX(unsigned int) - b.
 *
 * The main goal of this function is to minimize rounding errors when scaling any other radnomly generated floating point
 * numbers. Thus it takes uint bits directly as a source of randomness. If all bits are uniformly distributes
 * then scaling them to arbitrary floating point segment must provide uniform distribution of random floating point numbers
 * in a given range.
 *
 * Due to implementation details it is not recomended to pass $(D_PARAM a) and $(D_PARAM b) close to ±1.0e28 as that will
 * cause function to generate infinities.
 *
 * Params:
 *     ptr = Pointer to an array of random bits/resulting floating point values.
 *     a = Left bound of the segment.
 *     b = Right bound of the segment.
 *     count = Number of float values to scale.
 */
__global__
void kernel_scale(void *ptr, const float a, const float b, const size_t count)
{
	unsigned int *uPtr = (unsigned int*)ptr;
	float        *fPtr = (float*)ptr;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
		fPtr[i] = scale(uPtr[i], a, b);
}

/// ditto
__host__
void cuda_scale(void *ptr, const float a, const float b, const size_t count)
{
	kernel_scale<<<(count + 1023) / 1023, 1024>>>(ptr, a, b, count);
}

/**
 * BLX-α crossover.
 *
 * BLX-α crossover is used for real-coded problems. The idea is to pick a random offspring in the space between
 * two parents x and y extended by the <math><mi>α</mi></math> parameter. Offspring's genes are picked randomly in the range
 * <math><mrow><mfenced open="[" close="]" separators="; ">
 *     <mrow>
 *         <mtext>min</mtext><mfenced open="(" close=")" separators=", ">
 *             <msub><mi>X</mi><mi>i</mi></msub><msub><mi>Y</mi><mi>i</mi></msub>
 *         </mfenced>
 *         <mo>-</mo>
 *         <mi>α</mi><msub><mi>d</mi><mi>i</mi></msub>
 *     </mrow>
 *     <mrow>
 *         <mtext>max</mtext><mfenced open="(" close=")" separators=", ">
 *             <msub><mi>X</mi><mi>i</mi></msub><msub><mi>Y</mi><mi>i</mi></msub>
 *         </mfenced>
 *         <mo>+</mo>
 *         <mi>α</mi><msub><mi>d</mi><mi>i</mi></msub>
 *     </mrow>
 * </mfenced></mrow></math>
 * , where
 * <math><mrow>
 *     <mi>d</mi>
 *     <mo>=</mo>
 *     <mfenced open="|" close="|" separators="">
 *         <msub><mi>x</mi><mi>i</mi></msub><mo>-</mo><msub><mi>y</mi><mi>i</mi></msub>
 *     </mfenced>
 * </mrow></math>.
 * A picked value will not be out of the range
 * <math><mfenced open="[" close="]" separators="; "><mi>a</mi><mi>b</mi></mfenced></math> which one is limiting
 * the search space.
 *
 * Params:
 *     x = Parent array.
 *     y = Parent array.
 *     offspring = Offspring array.
 *     a = Minimal crossover value.
 *     b = Maximal crossover value.
 *     alpha = α parameter of BLX-α crossover. Must be &ge 0. Determines how much to extend the search space, where 0 means
 *         not to extend at all. 
 *     u = Pointer to an array of random bits. To prevent rounding errors it is of uint type rather than float value
 *         in range [0; 1]. These bits will be translated to float, where 0 translates to the left bound of the search space
 *         and uint.max - to the right bound.
 */
__global__
void kernel_BLX_a(
	const float *x, const float *y,
	float *offspring,
	const float a, const float b,
	const float alpha,
	const unsigned int *u,
	const size_t count
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
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
	const size_t count
)
{
	kernel_BLX_a<<<(count + 1023) / 1023, 1024>>>(x, y, offspring, a, b, alpha, u, count);
}

/**
 * Solve a quadratic equation.
 * 
 * Params:
 *     x1, x2 = Roots of the equation.
 *     a = x^2 coefficient.
 *     b = x coefficiant.
 *     c = Free coefficient. 
 */
__device__
void quadratic(float &x1, float &x2, const float a, const float b, const float c)
{
	const float D = powf(b, 2) - 4.0f * a * c;
	
	x1 = (-b - sqrtf(D)) / 2.0f / a;
	x2 = (-b + sqrtf(D)) / 2.0f / a;
}

/**
 * Rank based parent selection.
 *
 * Rank based selection is similar to roulette-wheel selection in which parents are selected with a probability
 * proportionate to their fitness values. Instead, in the rank based selection probabilities are proportionate
 * to the individual averall rank.
 *
 * This approach lets individuals with lower fitness to breed more often thus preserving genetic diversity and slowing down
 * convergence. This is especially notable with few individuals having fitness values much higher than the average
 * population. If parents are selected by the roulette-wheel selection, those best individuals will quicly take over all
 * population and solution will converge to fast to a local optimum. In the case of the rank based selection even the hugest
 * gap in fitness values will not speed up convergence and the global optimum will be searched better.
 *
 * Params:
 *     ranks = Ranks selected based on scores.
 *     scores = Array of scores.
 */
__global__
void kernel_RBS(unsigned int *ranks, const float *scores, const size_t count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
	{
		float x1, x2; // Equation roots
		
		quadratic(x1, x2, 0.5f, 0.5f, -scores[i]);
		
		float max_root = fmaxf(x1, x2);
		
		// ceilf(x - 1.0) is actually not equivalent to floorf(x) at integer values.
		// ceilf(2.0 - 1.0) = ceilf(1.0) = 1.0
		// floorf(2.0) = 2.0
		ranks[i] = (unsigned int)ceilf(max_root - 1.0f);
	}
}

__host__
void cuda_RBS(unsigned int *ranks, const float *scores, const size_t count)
{
	kernel_RBS<<<(count + 1023) / 1023, 1024>>>(ranks, scores, count);
}

/**
 * Fill the array x on a GPU with the value val.
 *
 * Params:
 *     x = A pointer to an array to fill.
 *     val = A value to fill with.
 *     n = Number of elements to fill.
 */
__global__
void kernel_fill(float *x, const float val, const size_t count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
		x[i] = val;
}

/// ditto
__host__
void cuda_fill(float *x, const float val, const size_t count)
{
	kernel_fill<<<(count + 1023) / 1023, 1024>>>(x, val, count);
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
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < count)
	{
		y[i] = 0;
		
		for (int j = 0; j < dim; ++j)
			y[i] += powf(x[i + count * j], 2);
		
		y[i] = sqrtf(y[i]);
	}
}

/// ditto
__host__
void cuda_L2(const float *x, float *y, const unsigned int dim, const size_t count)
{
	kernel_L2<<<(count + 1023) / 1023, 1024>>>(x, y, dim, count);
}

