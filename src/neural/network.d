/**
 * Copyright © 2017 - 2018 Sergei Iurevich Filippov, All Rights Reserved.
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
 * Neural network and related things.
 */
module neural.network;

// Standard D modules
import std.algorithm : all, each, swap;
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
	
	private
	{
		CurandGenerator curandGenerator;
		RandomPool      randomPool;
		cublasHandle_t  cublasHandle;
	}
	
	static this()
	{
		curandGenerator = CurandGenerator(curandRngType_t.PSEUDO_DEFAULT);
		randomPool      = RandomPool(curandGenerator);
		cublasCreate(cublasHandle);
	}
	
	static ~this()
	{
		curandGenerator.destroy;
		randomPool.freeMem();
		cublasDestroy(cublasHandle);
	}
}


immutable uint  biasLength = 1;   /// Number of bias weights per neuron.
immutable float biasWeight = 1.0; /// Weight of every bias connection.

enum LayerType { Input, Hidden, Output }; /// Layer types.

/**
 * Random layer generation parameters.
 */
struct LayerParams
{
	uint  inputs;  /// Number of layer input connections.
	uint  neurons; /// Number of neurons.

	float min = -1; /// Minimal generated weight.
	float max =  1; /// Maximal generated weight.
	
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
 * Random network generation parameters.
 */
struct NetworkParams
{
	uint  inputs;  /// Number of network's inputs.
	uint  outputs; /// Number of network's outputs.
	uint  neurons; /// Number of neurons in every hidden layer.
	uint  layers;  /// Number of hidden layers (excluding input and output layers).

	float min = -1; /// Minimal generated weight.
	float max =  1; /// Maximal generated weight.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (neurons >= 1);
		
		assert (max >= min);
		assert (isFinite(min));
		assert (isFinite(max));
	}
	
	/**
	 * Extract layer parameters from network parameters depending on a layer role in a net.
	 *
	 * Params:
	 *     type = Layer role in a network.
	 */
	@property LayerParams layerParams(in LayerType type) const @nogc nothrow pure @safe
	{
		LayerParams result;
		
		result.min = min;
		result.max = max;
		
		result.inputs  = (type == LayerType.Input  ? inputs  : neurons);
		result.neurons = (type == LayerType.Output ? outputs : neurons);
		
		return result;
	}
}

/**
 * Feedforward layer.
 *
 * Each neuron of this layer is connected to each neuron of the previous layer.
 */
struct Layer
{
	private Matrix weights; /// Connection weights.
	
	invariant
	{
		assert (&weights, "The weights matrix is incorrect.");
		assert (weights.rows >= 1 + biasLength); // connections()
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
	 *     generator = Pseudorandom number generator.
	 */
	this(in LayerParams params, CurandGenerator generator) nothrow @nogc
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
		mixin(writetest!__ctor);
		
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
	 * Evaluate the layer.
	 *
	 * Evaluates a result of feeding input matrix to the layer.
	 *
	 * Currently uses tanh() as activation function.
	 *
	 * Params:
	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
	 *     outputs = Output matrix of size m x n, where n is the number of neurons.
	 *     cublasHandle = Cublas handle.
	 *     activate = If set to true activation function will be applied to the result.
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
			cuda_tanh(outputs.ptr, outputs.length);
	}
	
	///
	unittest
	{
		mixin(writetest!opCall);
		
		immutable LayerParams params = { inputs : 2, neurons : 2 };
		
		Layer l = Layer(params, curandGenerator);
		scope(exit) l.freeMem();
		
		auto inputs = Matrix(4, 3);
		scope(exit) inputs.freeMem();
		
		auto outputs = Matrix(4, 2);
		scope(exit) outputs.freeMem();
		
		/*   Neurons
		 *   V    V
		 * 0.00 0.06 * <- weights
		 * 0.02 0.08 * <- weights
		 * 0.04 0.10 * <- biases */
		cudaDeviceSynchronize();
		l.weights.each!"a = i / 50f";
		
		inputs.each!"a = i";
		
		l(inputs, outputs, cublasHandle);
		cudaDeviceSynchronize();
		
		// cuBLAS matrices are column-major.
		immutable float[] result = [
			0.379949, 0.430084, 0.477700, 0.522665,
			0.807569, 0.876393, 0.921669, 0.950795
		];
		foreach (i, o; outputs)
			assert ( approxEqual(o, result[i], accuracy) );
	}
	
	/**
	 * Cross over parents to generate an offspring. The operation is performed on place.
	 *
	 * As the constructor allocates new memory for a new layer, to optimize performance and avoid memory reallocations
	 * this operation is performed in place assuming the calling struct is an offspring.
	 *
	 * Currently only BLX-α crossover is implemented and is a default algorithm.
	 *
	 * Params:
	 *     x = The first parent.
	 *     y = The second parent.
	 *     alpha = α parameter of BLX-α crossover. Simply put, determines how far to extend a search space from the parents
	 *         where 0 means not to extend at all. Generally, 0.5 is considered to show the best results.
	 *     pool = Pool of random numbers. It is supposed to improve performance of a crossover as cuRAND acheives maximum
	 *         efficiency generating large quontities of numbers.
	 */
	void crossover(in Layer x, in Layer y, in float alpha, RandomPool pool) nothrow @nogc
	in
	{
		assert (this.weights.length == y.weights.length, "Parents and an offspring must be the same size.");
		assert (this.weights.length == x.weights.length, "Parents and an offspring must be the same size.");
		assert (this.weights.length <= pool.size, "An offspring must not contain more values than a random pool does.");
		
		assert (alpha >= 0 && alpha <= 1, "α parameter must be in the range [0; 1]");
	}
	body
	{
		immutable length = this.weights.length;
		cuda_BLX_a(x.weights.ptr, y.weights.ptr, this.weights.ptr, alpha, pool(length).ptr, length);
	}
	
	///
	unittest
	{
		mixin(writetest!crossover);
		
		import std.algorithm : max, min;
		import std.math      : abs;
		
		immutable LayerParams params = {
			inputs  : 200,
			neurons : 300,
			min     : -1.0e30,
			max     : 2.0e31
		};
		
		immutable alpha = 0.5;
		
		auto parent1 = Layer(params, curandGenerator);
		scope(exit) parent1.freeMem();
		
		auto parent2 = Layer(params, curandGenerator);
		scope(exit) parent2.freeMem();
		
		auto offspring = Layer(params.inputs, params.neurons);
		scope(exit) offspring.freeMem();
		
		auto pool = RandomPool(curandGenerator, 1_000_000);
		scope(exit) pool.freeMem();
		
		offspring.crossover(parent1, parent2, alpha, pool);
		cudaDeviceSynchronize();
		
		foreach (i, off; offspring.weights)
			assert (
				off >= min(parent1.weights[i], parent2.weights[i]) - alpha * abs(parent1.weights[i] - parent2.weights[i]) &&
				off <= max(parent1.weights[i], parent2.weights[i]) + alpha * abs(parent1.weights[i] - parent2.weights[i])
			);
	}
}

/**
 * Simple feedforward neural network.
 */
struct Network
{
	Layer   inputLayer;   /// Input layer.
	Layer[] hiddenLayers; /// Hidden layers. It is possible to have no hidden layers at all.
	Layer   outputLayer;  /// Output layer.
	
	invariant
	{
		assert (
			hiddenLayers.all!(x => x.neuronsLength == inputLayer.neuronsLength),
			"Every hidden layer must have the same number of neurons as the input layer."
		);
	}
	
	/**
	 * Number of hidden layers (input and output does not count).
	 */
	@property ulong depth() const pure nothrow @safe @nogc
	{
		return hiddenLayers.length;
	}
	
	/**
	 * Get number of neurons per hidden layer.
	 */
	@property uint neuronsPerLayer() const pure nothrow @safe @nogc
	{
		return inputLayer.neuronsLength;
	}
	
	/**
	 * Constructor for random neural network.
	 *
	 * Params:
	 *     params = Network parameters.
	 *     generator = Pseudorandom number generator.
	 */
	this(in NetworkParams params, RandomPool pool) nothrow @nogc
	in
	{
		assert (&params, "Neural network parameters are incorrect.");
	}
	body
	{
		scope(failure) freeMem();
		
		inputLayer = Layer(
			params.layerParams(LayerType.Input),
			pool
		);
		outputLayer = Layer(
			params.layerParams(LayerType.Output),
			pool
		);
		
		hiddenLayers = nogcMalloc!Layer(params.layers);
		foreach (ref l; hiddenLayers)
			l = Layer(
				params.layerParams(LayerType.Hidden),
				pool
			);
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		NetworkParams params = { layers : 2, inputs : 2, neurons : 3, outputs : 1 };
		
		Network network = Network(params, randomPool);
		scope(exit) network.freeMem();
		
		with (network)
		{
			assert (depth           == params.layers);
			assert (neuronsPerLayer == params.neurons);
			
			with (inputLayer)
			{
				assert (connectionsLength == params.inputs + biasLength);
				assert (neuronsLength     == params.neurons);
				
				assert (weights.values.all!(x => isFinite(x)));
				assert (weights.values.all!(x => x >= params.min && x <= params.max));
			}
			
			with (outputLayer)
			{
				assert (connectionsLength == params.neurons + biasLength);
				assert (neuronsLength     == params.outputs);
				
				assert (weights.values.all!(x => isFinite(x)));
				assert (weights.values.all!(x => x >= params.min && x <= params.max));
			}
			
			assert (hiddenLayers.all!(x => x.connectionsLength == params.neurons + biasLength));
			assert (hiddenLayers.all!(x => x.neuronsLength == params.neurons));
			
			assert (
				hiddenLayers.all!(
					l => l.weights.values.all!(
						x => isFinite(x)
			)));
			assert (
				hiddenLayers.all!(
					l => l.weights.values.all!(
						x => x >= params.min && x <= params.max
			)));
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
		inputLayer.freeMem();
		outputLayer.freeMem();
		
		foreach (ref l; hiddenLayers)
			l.freeMem();
		
		free(hiddenLayers);
	}
	
	/**
	 * Evaluate the layer.
	 *
	 * Evaluates a result of feeding inpit matrix to the network.
	 *
	 * Params:
	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
	 *     outputs = Output matrix of size m x n, where n is the number of output neurons.
	 *     cublasHandle = Cublas handle.
	 */
	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle) const nothrow @nogc
	{
		auto prev = Matrix(inputs.rows, neuronsPerLayer + biasLength);
		scope(exit) prev.freeMem();
		
		auto next = Matrix(inputs.rows, neuronsPerLayer + biasLength);
		scope(exit) next.freeMem();
		
		cuda_fill(prev.colSlice(prev.cols - 1, prev.cols).ptr, biasWeight, prev.rows);
		cuda_fill(next.colSlice(next.cols - 1, next.cols).ptr, biasWeight, next.rows);
		
		inputLayer(inputs, prev.colSlice(0, prev.cols - 1), cublasHandle);
		foreach (l; hiddenLayers)
		{
			l(prev, next.colSlice(0, next.cols - 1), cublasHandle);
			swap(prev, next);
		}
		outputLayer(prev, outputs, cublasHandle, false);
	}
	
	///
	unittest
	{
		mixin(writetest!opCall);
		
		immutable NetworkParams params = {
			inputs  : 2,
			outputs : 1,
			neurons : 3,
			layers  : 3
		};
		
		immutable measures = 4;
		
		auto inputs = Matrix(measures, params.neurons);
		scope(exit) inputs.freeMem();
		
		auto outputs = Matrix(measures, params.outputs);
		scope(exit) outputs.freeMem();
		
		auto network = Network(params, curandGenerator);
		scope(exit) network.freeMem();
		
		copy(
			[ 1.0,        2.0,        3.0,        4.0,
			  5.0,        6.0,        7.0,        8.0,
			  biasWeight, biasWeight, biasWeight, biasWeight ], // column for bias
			inputs
		);
		
		copy( //        bias
			[ 0.1, 0.2, 0.3,   // <- 1st neuron
			  0.4, 0.5, 0.6,   // <- 2nd neuron
			  0.7, 0.8, 0.9 ], // <- 3rd neuron
			network.inputLayer.weights
		);
		
		copy( //                bias
			[ -0.1, -0.2, -0.3, -0.4,   // <- 1st neuron
			  -0.5, -0.6, -0.7, -0.8,   // <- 2nd neuron
			  -0.9, -1.0, -1.1, -1.2 ], // <- 3rd neuron
			network.hiddenLayers[0].weights
		);
		
		copy( //             bias
			[ 0.1, 0.2, 0.3, 0.4,   // <- 1st neuron
			  0.5, 0.6, 0.7, 0.8,   // <- 2nd neuron
			  0.9, 1.0, 1.1, 1.2 ], // <- 3rd neuron
			network.hiddenLayers[1].weights
		);
		
		copy( //                bias
			[ -0.1, -0.2, -0.3, -0.4,   // <- 1st neuron
			  -0.5, -0.6, -0.7, -0.8,   // <- 2nd neuron
			  -0.9, -1.0, -1.1, -1.2 ], // <- 3rd neuron
			network.hiddenLayers[2].weights
		);
		
		copy(
			[ 0, 1, 2, 3 ], // the only neuron
			network.outputLayer.weights
		);
		
		network(inputs, outputs, cublasHandle);
		cudaDeviceSynchronize();
		
		immutable float[] result = [4.497191, 4.500117, 4.501695, 4.502563];
		foreach (i, o; outputs)
			assert ( approxEqual(o, result[i], accuracy) );
	}
}

