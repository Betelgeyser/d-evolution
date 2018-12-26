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
import math.matrix : Matrix;

version (unittest)
{
	import common;
	
	import std.algorithm : each, equal;
	import std.math      : approxEqual;
}


private extern (C++) void cuda_tanh(float* x, const size_t n) nothrow @nogc;
/**
 * Calculate hyperbolic tangent of each element of an array $(D_PARAM x) on a GPU in place.
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
 *     x = Array to calculate.
 */
void cudaTanh(float[] x) @nogc nothrow
{
	cuda_tanh(x.ptr, x.length);
}

///
unittest
{
	mixin(writeTest!cudaTanh);
	
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

private extern (C++) void cuda_ReLU(float* x, const size_t n) @nogc nothrow;
/**
 * Calculate rectified linear unit of each element of an array $(D_PARAM x) on a GPU in place.
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
 *     x = Array to calculate.
 */
void cudaReLU(float[] x) nothrow @nogc
{
	cuda_ReLU(x.ptr, x.length);
}

///
unittest
{
	mixin(writeTest!cudaReLU);
	
	immutable length = 3;
	
	float[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	data[0 .. $] = [-1, 0, 1];
	
	cudaReLU(data);
	cudaDeviceSynchronize();
	
	immutable float[] result = [0, 0, 1];
	assert (equal!approxEqual(data, result));
}

private extern (C++) void cuda_LeakyReLU(float* x, const size_t n) @nogc nothrow;
/**
 * Calculate leaky rectified linear unit of each element of an array $(D_PARAM x) on a GPU in place.
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
 *     x = Array to calculate.
 */
void cudaLeakyReLU(float[] x) nothrow @nogc
{
	cuda_LeakyReLU(x.ptr, x.length);
}

///
unittest
{
	mixin(writeTest!cudaLeakyReLU);
	
	immutable length = 3;
	
	float[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	data[0 .. $] = [-1, 0, 1];
	
	cudaLeakyReLU(data);
	cudaDeviceSynchronize();
	
	immutable float[] result = [-0.01, 0, 1];
	assert (equal!approxEqual(data, result));
}

private extern (C++) void cuda_softPlus(float* x, const size_t n) @nogc nothrow;
/**
 * Calculate softplus function of each element of an array $(D_PARAM x) on a GPU in place.
 *
 * <math><mrow>
 *     <mi mathvariant="italic">SoftPlus</mi><mo>(</mo><mi>x</mi><mo>)</mo>
 *     <mo>=</mo>
 *     <mi mathvariant="italic">ln</mi><mo>(</mo><mn>1</mn><mo>-</mo><msup><mi>e</mi><mi>x</mi></msup><mo>)</mo>
 * </mrow></math>
 *
 * Params:
 *     x = Array to calculate.
 */
void cudaSoftPlus(float[] x) nothrow @nogc
{
	cuda_softPlus(x.ptr, x.length);
}

///
unittest
{
	mixin(writeTest!cudaSoftPlus);
	
	immutable length = 7;
	
	float[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	data[0 .. $] = [-1000, -10, -1, 0, 1, 10, 1000];
	
	cudaSoftPlus(data);
	cudaDeviceSynchronize();
	
	immutable float[] result = [0.000000, 0.000045, 0.313261, 0.693147, 1.313261, 10.000045, 1000.000000];
	assert (equal!approxEqual(data, result));
}

private extern (C++) void cuda_scale(void* ptr, const float  a, const float  b, const size_t count) nothrow @nogc;
/**
 * Transform uniformly distrubuted random bits into uniformly distributed random floating point numbers in range 
 * [$(D_PARAM a); $(D_PARAM b)], where $(D_PARAM a) &le; $(D_PARAM b). 0 will translate to $(D_PARAM a) and uint.max
 * to $(D_PARAM b).
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
 *     x = Array of random bits.
 *     a = Left bound of the segment.
 *     b = Right bound of the segment.
 *
 * Returns: A new pointer to the array $(D_PARAM x) of float type.
 */
float[] cudaScale(uint[] x, in float a, in float b) nothrow @nogc
in
{
	assert (a <= b);
}
body
{
	cuda_scale(x.ptr, a, b, x.length);
	return cast(float[])x;
}

///
unittest
{
	mixin(writeTest!cudaScale);
	
	immutable length = 3;
	
	uint[] data;
	cudaMallocManaged(data, length);
	scope(exit) cudaFree(data);
	
	data[0 .. $]   = [uint.min, uint.max / 2, uint.max];
	float[] result = [      -1,            0,        1];
	
	// fData is just a copy pointer
	float[] fData = cudaScale(data, -1, 1);
	cudaDeviceSynchronize();
	
	assert(equal!approxEqual(fData, result));
	
	data[0 .. $] = [uint.min, uint.max / 2,  uint.max];
	result       = [       0,      500_000, 1_000_000];
	
	fData = cudaScale(data, 0, 1_000_000);
	cudaDeviceSynchronize();
	
	assert(equal!approxEqual(fData, result));
}

private extern (C++) void cuda_BLX_a(
	const(float*) x, const(float*) y,
	float* offspring,
	const float a, const float b,
	const float alpha,
	const(uint*) u,
	const size_t n
) nothrow @nogc;
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
 *         not to extend at all. Apperantly it is considered 0.5 shows the best results.
 *     u = Pointer to an array of random bits. To prevent rounding errors it is of uint type rather than float value
 *         in range [0; 1]. These bits will be translated to float, where 0 translates to the left bound of the search space
 *         and uint.max - to the right bound.
 */
void cudaBLXa(in float[] x, in float[] y, float[] offspring, in float a, in float b, const float alpha, in uint[] u) nothrow @nogc
in
{
	assert (offspring.length == x.length);
	assert (offspring.length == y.length);
	assert (offspring.length == u.length);
	
	assert (alpha >= 0);
}
body
{
	cuda_BLX_a(x.ptr, y.ptr, offspring.ptr, a, b, alpha, u.ptr, offspring.length);
}

///
unittest
{
	mixin(writeTest!cudaBLXa);
	
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
	
	// There should be pregenerated random values
	uint[] u;
	cudaMallocManaged(u, length);
	scope(exit) cudaFree(u);
	
	u[0 .. $] = [0, uint.max / 2, uint.max];
	
	// Artificial crossover. u will be random in real calculations.
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

private extern(C++) void cuda_RBS(uint* ranks, const float* scores, const size_t count) @nogc nothrow;
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
void cudaRBS(uint[] ranks, in float[] scores) @nogc nothrow
in
{
	assert (ranks.length == scores.length);
}
body
{
	cuda_RBS(ranks.ptr, scores.ptr, ranks.length);
}

///
unittest
{
	mixin(writeTest!cudaRBS);
	
	uint[] ranks;
	cudaMallocManaged(ranks, 5);
	scope(exit) cudaFree(ranks);
	
	float[] scores;
	cudaMallocManaged(scores, 5);
	scope(exit) cudaFree(scores);
	
	scores[0 .. $] = [0.0, 0.1, 3.0, 4.5, 47.123];
	
	cudaRBS(ranks, scores);
	cudaDeviceSynchronize();
	
	immutable uint[] result = [0, 0, 1, 2, 9];
	assert (equal(ranks, result));
}

private extern(C++) void cuda_fill(float* x, const float val, const size_t n) nothrow @nogc;
/**
 * Fill the array $(D_PARAM x) on a GPU with the value $(D_PARAM val).
 *
 * Params:
 *     x = An array to fill.
 *     val = A value to fill with.
 */
void cudaFill(float[] x, in float val) nothrow @nogc
{
	cuda_fill(x.ptr, val, x.length);
}

///
unittest
{
	mixin(writeTest!cudaFill);
	
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

private extern(C++) void cuda_L2(const(float)* x, float* y, const uint dim, const size_t count) nothrow @nogc;
/**
 * Per-vector calculation of the Euclidean distance (L2 norm) of a vector array on a GPU.
 *
 * Params:
 *     x = An array of vectors which is represented in a matrix form for convenience. Each column is a vector.
 *         Thus, each row is a dimention.
 *     y = A resulting array of L2 norm values. Its length must equals to number of the columns in the input matrix.
 */
void cudaL2(in Matrix x, float[] y) nothrow
{
	if (x.rows != y.length)
		throw new Error("Input and output arrays have different sizes.");
	
	cuda_L2(x.ptr, y.ptr, x.cols, x.rows);
}

///
unittest
{
	mixin(writeTest!cudaL2);
	
	immutable dim    = 4;
	immutable length = 2;
	
	Matrix data = Matrix(length, dim);
	scope(exit) data.freeMem();
	
	float[] norm;
	cudaMallocManaged(norm, length);
	scope(exit) cudaFree(norm);
	
	data.values.each!"a = i";
	
	cudaL2(data, norm);
	cudaDeviceSynchronize();
	
	immutable float[] result = [7.483315, 9.165151];
	assert (equal!approxEqual(norm, result));
}

