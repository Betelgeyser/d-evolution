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
import std.format;
import std.math      : isFinite;
import std.algorithm : each, all, copy, swap;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import common;
import math;

version (unittest)
{
	import std.math : approxEqual;
	immutable accuracy = 0.000_001;
	
	CurandGenerator curandGenerator;
	cublasHandle_t  cublasHandle;
		 
	static this()
	{
		curandGenerator = CurandGenerator(curandRngType_t.PSEUDO_DEFAULT);
		cublasCreate(cublasHandle);
	}
	
	static ~this()
	{
		curandGenerator.destroy;
		cublasDestroy(cublasHandle);
	}
}


immutable biasLength = 1; /// Number of bias weights per neuron.
immutable biasWeight = 1; /// Weight of every bias connection.

enum LayerType { Input, Hidden, Output }; /// Layer types.

/**
 * Random layer generation parameters.
 */
struct LayerParams
{
	uint  inputs;  /// Number of layer input connections.
	uint  neurons; /// Number of neurons.

	float min = -float.max; /// Minimal generated weight.
	float max =  float.max; /// Maximal generated weight.
	
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

	float min = -float.max; /// Minimal generated weight.
	float max =  float.max; /// Maximal generated weight.
	
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
	LayerParams getLayerParams(in LayerType type) const pure nothrow @safe @nogc
	{
		LayerParams result;
		
		result.min = min;
		result.min = max;
		
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
	 * Constructor without initialization.
	 *
	 * Params:
	 *     inputs = Number of weights per neuron.
	 *     neurons = Number of neurons in the layer.
	 */
	this(in uint inputs, in uint neurons) nothrow @nogc
	in
	{
		assert (inputs  >= 1);
		assert (neurons >= 1);
	}
	body
	{
		scope(failure) freeMem();
		
		weights = Matrix(inputs + biasLength, neurons);
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
		this(params.inputs, params.neurons);
		
		{
			scope(failure) freeMem();
			generator.generateUniform(weights.ptr, length);
			cuda_scale(weights.ptr, params.min, params.max, length);
		}
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		import std.math : isFinite;
		
		
		immutable LayerParams params = { inputs : 200, neurons : 300, min : -1.0e30, max : 2.0e31 };
		
		auto l = Layer(params, curandGenerator);
		scope(exit) l.freeMem();
		cudaDeviceSynchronize();
		
		assert (l.connectionsLength == params.inputs + biasLength);
		assert (l.neuronsLength     == params.neurons);
		
		assert ( l.weights.values.all!(x => isFinite(x)) );
		assert ( l.weights.values.all!(x => x >= params.min) );
		assert ( l.weights.values.all!(x => x <= params.max) );
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
	 * Evaluates a result of feeding inpit matrix to the layer.
	 *
	 * Currently uses tanh() as activation function.
	 *
	 * Although sizes of matricies are checked before multiplication, output matrix is allowed to have more columns than
	 * there is neurons in the layer. This restriction is omited to make possible to pass output matricies with additional
	 * column to multiply bias of the netx layer on.
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
		assert (inputs.cols    == connectionsLength, "The number of matrix columns must be equal to the layer's connections number.");
		assert (inputs.rows    == outputs.rows, "The input and the output matrix must have the same number of rows.");
		assert (neuronsLength  <= outputs.cols, "The output matrix must not have less columns than the layer's neurons number.");
	}
	body
	{
		gemm(inputs, weights, outputs, cublasHandle);
		
		// TODO: need extended matrix here. Extended part should not be activated.
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
		
		immutable LayerParams params = { inputs : 200, neurons : 300, min : -1.0e30, max : 2.0e31 };
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
	Layer[] hiddenLayers; /// Hidden layers. It is possible to have 0 hidden layers.
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
	this(in NetworkParams params, CurandGenerator generator) nothrow @nogc
	in
	{
		assert (&params, "Neural network parameters are incorrect.");
	}
	body
	{
		scope(failure) freeMem();
		
		inputLayer = Layer(
			params.getLayerParams(LayerType.Input),
			generator
		);
		outputLayer = Layer(
			params.getLayerParams(LayerType.Output),
			generator
		);
		
		hiddenLayers = nogcMalloc!Layer(params.layers);
		foreach (ref l; hiddenLayers)
			l = Layer(
				params.getLayerParams(LayerType.Hidden),
				generator
			);
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		NetworkParams params;
		params.inputs  = 2;
		params.layers  = 2;
		params.neurons = 3;
		params.outputs = 1;
		
		
		Network n = Network(params, generator); scope(exit) n.freeMem();
		cudaDeviceSynchronize();
		
		assert (n.depth           == params.layers);
		assert (n.neuronsPerLayer == params.neurons);
		
		assert (n.inputLayer.connectionsLength == params.inputs + biasLength);
		assert (n.inputLayer.neuronsLength     == params.neurons);
		
		assert (n.outputLayer.connectionsLength == params.neurons + biasLength);
		assert (n.outputLayer.neuronsLength     == params.outputs);
		
		assert ( n.hiddenLayers.all!(x => x.connectionsLength == params.neurons + biasLength) );
		assert ( n.hiddenLayers.all!(x => x.neuronsLength     == params.neurons) );
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
		
		if (depth > 0)
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
		immutable extensionOffset = inputs.cols * neuronsPerLayer;
		
		auto prev = Matrix(inputs.cols, neuronsPerLayer + biasLength);
		auto next = Matrix(inputs.cols, neuronsPerLayer + biasLength);
		
		inputLayer(inputs, prev, cublasHandle);
		cuda_fill(prev + extensionOffset, biasWeight, neuronsPerLayer + biasLength);
		
		foreach (l; hiddenLayers)
		{
			l(prev, next, cublasHandle);
			prev = next;
			
			// TODO: need extemded matrix instead. Extended part should not be activated.
			cuda_fill(prev + extensionOffset, biasWeight, neuronsPerLayer + biasLength);
		}
		
		outputLayer(prev, outputs, cublasHandle, false);
	}
	
	///
	unittest
	{
		mixin(writetest!opCall);
		
		
		NetworkParams params;
		params.inputs  = 1;
		params.outputs = 1;
		params.neurons = 1;
		params.layers  = 1;
		
		auto inputs = Matrix(2, 2);
		inputs.values[0] = 0;
		inputs.values[1] = 1;
		inputs.values[2] = 1;
		inputs.values[3] = 1;
		
		auto outputs = Matrix(2, 1);
		
		/* This is how test network works for 1st input value: *
		 *                                                     *
		 * 0 - 0.00 - (in) - 2.00 - (hn) - -0.50 - (on) ---- o *
		 *           /             /              /         /  *
		 *   1 - 1.00     1 - -2.00       1 - 0.75  0.971843   */
		auto network = Network(params, generator);
		cudaDeviceSynchronize();
		network.inputLayer.weights[0]      =  0.00;
		network.inputLayer.weights[1]      =  1.00; // bias
		network.hiddenLayers[0].weights[0] =  2.00;
		network.hiddenLayers[0].weights[1] = -2.00; // bias
		network.outputLayer.weights[0]     = -0.50;
		network.outputLayer.weights[1]     =  0.75; // bias
		
		network(inputs, outputs, handle);
		cudaDeviceSynchronize();
		
		float[] result = [0.971843, 0.971843];
		for (ulong i = 0; i < outputs.length; ++i)
			assert (
				approxEqual(
					outputs[i], result[i],
					accuracy
				)
			);
	}
}

