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
module neural.layer;

// Standard D modules
import std.algorithm : all, each;
import std.math      : isFinite;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import common;
import math;

version (unittest)
{
	import std.algorithm : equal;
	import std.math      : approxEqual;
	
	private cublasHandle_t  cublasHandle;
	
	static this()
	{
		cublasCreate(cublasHandle);
	}
	
	static ~this()
	{
		cublasDestroy(cublasHandle);
	}
}


immutable uint  biasLength = 1;   /// Number of bias weights per neuron.
immutable float biasWeight = 1.0; /// Weight of every bias connection.

/**
 * Random layer generation parameters.
 */
struct LayerParams
{
	uint  inputs;  /// Number of layer input connections.
	uint  neurons; /// Number of neurons.

	float min = -1; /// Minimum generated weight.
	float max =  1; /// Maximum generated weight.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (neurons >= 1);
		
		assert (max >= min);
		assert (isFinite(min));
		assert (isFinite(max));
	}
}

/**
 * Feedforward layer.
 *
 * Each neuron of this layer is connected to each neuron of the previous layer.
 */
struct Layer
{
	Matrix weights; /// Connections' weights.
	
	invariant
	{
		assert (&_weights, "The weights matrix is incorrect.");
		assert (_weights.rows >= 1 + biasLength,
			"The weights matrix must have at least `biasLength + 1` rows. 1 comes for a neuron.");
	}
	
	/**
	 * Number of connections per neuron (including bias).
	 */
	@property uint connectionsLength() const pure nothrow @safe @nogc
	{
		return weights.rows;
	}
	
	/**
	 * Number of neurons in the layer.
	 */
	@property uint neuronsLength() const pure nothrow @safe @nogc
	{
		return weights.cols;
	}
	
	/**
	 * Total number of weights.
	 */
	@property ulong length() const pure nothrow @safe @nogc
	{
		return weights.length;
	}
	
	/**
	 * Consctroctor with random initialization.
	 *
	 * Params:
	 *     params = Layer parameters.
	 *     pool = Pseudorandom number generator.
	 */
	this(in LayerParams params, RandomPool pool) nothrow @nogc
	in
	{
		assert (&params, "Incorrect layer parameters.");
	}
	body
	{
		scope(failure) freeMem();
		
		weights = Matrix(params.inputs + biasLength, params.neurons);
		
		auto tmpPtr = cudaScale(pool(length), params.min, params.max)[0 .. $];
		cudaDeviceSynchronize();
		weights.values[0 .. $] = tmpPtr[0 .. $];
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		immutable LayerParams params = { inputs : 20, neurons : 30 };
		
		auto layer = Layer(params, randomPool);
		scope(exit) layer.freeMem();
		
		with (layer)
		{
			assert (connectionsLength == params.inputs + biasLength);
			assert (neuronsLength     == params.neurons);
			
			assert (
				weights.values.all!(
					x => isFinite(x)
			));
			
			assert (
				weights.values.all!(
					x => x >= params.min && x <= params.max
			));
		}
	}
	
	/**
	 * Free memory.
	 *
	 * For the reason how D works with structs memory freeing moved from destructor to
	 * the the distinct function. Either allocating structs on stack or in heap or both
	 * causes spontaneous destructors calls. Apparently structs are not intended
	 * to be used with dynamic memory, probably it should be implemented as a class.  
	 */
	void freeMem() nothrow @nogc
	{
		weights.freeMem();
	}
	
	/**
	 * Activate the layer.
	 *
	 * Calculates a result of feeding an input matrix to the layer.
	 *
	 * Currently uses `tanh()` as an activation function.
	 *
	 * Params:
	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
	 *     outputs = Output matrix of size m x n, where n is the number of neurons.
	 *     cublasHandle = Cublas handle.
	 *     activate = If set to `true` activation function will be applied to the result.
	 */
	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle, in bool activate = true) const nothrow @nogc
	in
	{
		assert (inputs.cols   == connectionsLength);
		assert (inputs.rows   == outputs.rows);
		assert (neuronsLength == outputs.cols);
	}
	body
	{
		gemm(inputs, weights, outputs, cublasHandle);
		
		if (activate)
			cudaTanh(outputs);
	}
	
	///
	unittest
	{
		mixin(writeTest!opCall);
		
		immutable LayerParams params = { inputs : 2, neurons : 2 };
		
		Layer layer = Layer(params, randomPool);
		scope(exit) layer.freeMem();
		
		auto inputs = Matrix(4, 3);
		scope(exit) inputs.freeMem();
		
		auto outputs = Matrix(4, 2);
		scope(exit) outputs.freeMem();
		
		/*   Neurons
		 *   V    V
		 * 0.00 0.06 * <- weights
		 * 0.02 0.08 * <- weights
		 * 0.04 0.10 * <- biases */
		layer.weights.each!"a = i / 50f";
		
		inputs.each!"a = i";
		
		layer(inputs, outputs, cublasHandle);
		cudaDeviceSynchronize();
		
		// cuBLAS matrices are column-major.
		immutable float[] result = [
			0.379949, 0.430084, 0.477700, 0.522665,
			0.807569, 0.876393, 0.921669, 0.950795
		];
		assert (equal!approxEqual(outputs, result));
	}
	
	/**
	 * Cross over parents to generate an offspring. The operation is performed in place.
	 *
	 * As the constructor allocates new memory for a new layer, to optimize performance and avoid memory reallocations
	 * this operation is performed in place assuming the calling struct is an offspring.
	 *
	 * Currently only BLX-α crossover is implemented and this is a default algorithm.
	 *
	 * From more details look $(LINK2 ../math/kernels.html#cudaBLXa,math.kernels.cudaBLXa).
	 *
	 * Params:
	 *     x = The first parent.
	 *     y = The second parent.
	 *     a = Minimal crossover value.
	 *     b = Maximal crossover value.
	 *     alpha = α parameter of BLX-α crossover.
	 *     pool = Pool of random bits. It is supposed to improve performance of a crossover as the cuRAND acheives maximum
	 *         efficiency generating large quantities of numbers.
	 */
	void crossover(in Layer x, in Layer y, in float a, in float b, in float alpha, RandomPool pool) nothrow @nogc
	in
	{
		assert (this.length == y.length);
		assert (this.length == x.length);
		
		assert (a <= b);
		
		assert (alpha >= 0, "α parameter must be >= 0");
		
		assert (this.length <= pool.length, "RandomPool must contain at least as much numbers as a layer does.");
	}
	body
	{
		cudaBLXa(x.weights, y.weights, weights, a, b, alpha, pool(length));
	}
	
	///
	unittest
	{
		mixin(writeTest!crossover);
		
		import std.algorithm : max, min;
		import std.math      : abs;
		
		immutable LayerParams params = { inputs : 200, neurons : 300 };
		
		immutable alpha = 0.5;
		
		auto parent1 = Layer(params, randomPool);
		scope(exit) parent1.freeMem();
		
		auto parent2 = Layer(params, randomPool);
		scope(exit) parent2.freeMem();
		
		auto offspring = Layer(params, randomPool);
		scope(exit) offspring.freeMem();
		
		offspring.crossover(parent1, parent2, params.min, params.max, alpha, randomPool);
		cudaDeviceSynchronize();
		
		assert (
			offspring.weights.values.all!(
				x => isFinite(x)
		));
		
		foreach (i, off; offspring.weights)
		{
			float _min = min(parent1.weights[i], parent2.weights[i], params.min)
				- alpha * abs(parent1.weights[i] - parent2.weights[i]);
			
			float _max = max(parent1.weights[i], parent2.weights[i], params.max)
				+ alpha * abs(parent1.weights[i] - parent2.weights[i]);
			
			assert (off >= _min && off <= _max);
		}
	}
}

