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
 */
module neural.layer;

// Standard D modules
import std.algorithm : all, each;
import std.conv      : to;
import std.json      : JSONType, JSONValue;
import std.math      : isFinite;
import std.string    : format;

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
	import std.json      : parseJSON, toJSON;
	import std.math      : approxEqual;
	
	private cublasHandle_t cublasHandle;
	private RandomPool     randomPool;
	
	static this()
	{
		randomPool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, 0, 100_000);
		cublasCreate(cublasHandle);
	}
	
	static ~this()
	{
		cublasDestroy(cublasHandle);
		randomPool.freeMem();
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
	
	float min = -1.0; /// Minimum generated weight.
	float max =  1.0; /// Maximum generated weight.
	
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
	private Matrix _weights; /// The matrix that stores values of the neurons' weights.
	
	invariant
	{
	}
	
	/**
	 * Returns: The number of input parameters the layer takes, excluding bias.
	 */
	@property uint inputs() const @nogc nothrow pure @safe
	{
		return _weights.rows - biasLength;
	}
	
	/**
	 * Returns: The number of neurons in the layer.
	 */
	@property uint neurons() const @nogc nothrow pure @safe
	{
		return _weights.cols;
	}
	
	/**
	 * Returns: Total number of weights.
	 */
	@property ulong length() const @nogc nothrow pure @safe
	{
		return _weights.length;
	}
	
	/**
	 * Returns: The weights array of the layer.
	 */
	@property inout(float[]) weights() inout @nogc nothrow pure @safe
	{
		return _weights.values;
	}
	
	/**
	 * Returns: The size of the memory in bytes.
	 */
	@property size_t size() const @nogc nothrow pure @safe
	{
		return _weights.size;
	}
	
	/**
	 * Consctroctor with random initialization.
	 *
	 * Params:
	 *     params = Layer parameters.
	 *     pool = Pseudorandom number generator.
	 */
	this(in LayerParams params, RandomPool pool) nothrow
	{
		scope(failure) freeMem();
		
		_weights = Matrix(params.inputs + biasLength, params.neurons);
		
		auto tmpPtr = cudaScale(pool(length), params.min, params.max);
		cudaDeviceSynchronize();
		
		_weights.values[] = tmpPtr[];
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		immutable LayerParams params = { inputs : 5, neurons : 10 };
		
		auto layer = Layer(params, randomPool);
		scope(exit) layer.freeMem();
		
		with (layer)
		{
			assert (inputs  == params.inputs);
			assert (neurons == params.neurons);
			assert (length  == (params.inputs + biasLength) * params.neurons);
			
			assert (weights.all!(x => isFinite(x)));
			assert (weights.all!(x => x >= params.min && x <= params.max));
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
	void freeMem() nothrow
	{
		_weights.freeMem();
	}
	
	/**
	 * Deep copy.
	 *
	 * Params:
	 *     src = Layer to copy.
	 *     dst = Destination layer.
	 */
	static void copy(in Layer src, Layer dst) nothrow pure @safe
	{
		Matrix.copy(src._weights, dst._weights);
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
	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle, in bool activate = true) const nothrow
	in
	{
		assert (inputs.cols   == connectionsLength);
		assert (inputs.rows   == outputs.rows);
		assert (neuronsLength == outputs.cols);
	}
	body
	{
		gemm(inputs, false, _weights, false, outputs, cublasHandle);
		
		if (activate)
//			cudaTanh(outputs);
//			cudaReLU(outputs);
//			cudaSoftPlus(outputs);
			cudaLeakyReLU(outputs.values);
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
		
		//                        weight weight bias
		layer._weights[0 .. $] = [ 0.00,  0.02,  0.04,   // 1st neuron
		                           0.06,  0.08,  0.10 ]; // 2nd neuron
		
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
	void crossover(in Layer x, in Layer y, in float a, in float b, in float alpha, RandomPool pool) nothrow
	{
		if (a > b)
			throw new Error("Invalid crossover boundaries [%g; %g].".format(a, b));
		
		if (alpha < 0)
			throw new Error("α parameter must be ≥ 0, got %g.".format(alpha));
		
		if (this.length != x.length || this.length != y.length)
			throw new Error(
				"Child layer must have the same lenght as parents, got %d (child), %d and %d (parents)"
				.format(this.length, x.length, y.length)
			);
		
		cudaBLXa(x.weights, y.weights, _weights, a, b, alpha, pool(length));
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
		
		with (offspring)
		{
			assert (weights.all!(x => isFinite(x)));
			
			foreach (i, w; weights)
			{
				float diff = abs(parent1.weights[i] - parent2.weights[i]);
				
				float _min = min(parent1.weights[i], parent2.weights[i]);
				float _max = max(parent1.weights[i], parent2.weights[i]);
				
				_min -= alpha * diff;
				_max += alpha * diff;
				
				_min = max(_min, params.min);
				_max = min(_max, params.max);
				
				assert (w >= _min && w <= _max);
			}
		}
	}
}

