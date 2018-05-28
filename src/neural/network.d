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
import neural.layer;


immutable uint  biasLength = 1;   /// Number of bias weights per neuron.
immutable float biasWeight = 1.0; /// Weight of every bias connection.

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
	
	@property LayerParams inputParams() const @nogc nothrow pure @safe
	{
		LayerParams result = { min : this.min, max : this.max, inputs : this.inputs, neurons : this.neurons };
		return result;
	}
	
	@property LayerParams hiddenParams() const @nogc nothrow pure @safe
	{
		LayerParams result = { min : this.min, max : this.max, inputs : this.neurons, neurons : this.neurons };
		return result;
	}
	
	@property LayerParams outputParams() const @nogc nothrow pure @safe
	{
		LayerParams result = { min : this.min, max : this.max, inputs : this.neurons, neurons : this.outputs };
		return result;
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
	 *     pool = Pseudorandom number generator.
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
		
		hiddenLayers.each!(x => x.freeMem());
		
		if (hiddenLayers.length)
			free(hiddenLayers);
	}
	
	/**
	 * Activate the network.
	 *
	 * Claculates the result of feeding an inpit matrix to the network.
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
		
		cudaFill(prev.colSlice(prev.cols - 1, prev.cols), biasWeight);
		cudaFill(next.colSlice(next.cols - 1, next.cols), biasWeight);
		
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
		
		immutable NetworkParams params = { inputs : 2, outputs : 1, neurons : 3, layers : 3 };
		
		immutable measures = 4;
		
		auto inputs = Matrix(measures, params.neurons);
		scope(exit) inputs.freeMem();
		
		auto outputs = Matrix(measures, params.outputs);
		scope(exit) outputs.freeMem();
		
		auto network = Network(params, randomPool);
		scope(exit) network.freeMem();
		
		inputs[0 .. $] = [ 1.0,        2.0,        3.0,        4.0,
		                   5.0,        6.0,        7.0,        8.0,
		                   biasWeight, biasWeight, biasWeight, biasWeight ]; // column for bias
		
		//                                               bias
		network.inputLayer.weights[0 .. $] = [ 0.1, 0.2, 0.3,   // <- 1st neuron
		                                       0.4, 0.5, 0.6,   // <- 2nd neuron
		                                       0.7, 0.8, 0.9 ]; // <- 3rd neuron
		
		//                                                            bias
		network.hiddenLayers[0].weights[0 .. $] = [ -0.1, -0.2, -0.3, -0.4,   // <- 1st neuron
		                                            -0.5, -0.6, -0.7, -0.8,   // <- 2nd neuron
		                                            -0.9, -1.0, -1.1, -1.2 ]; // <- 3rd neuron
		
		//                                                         bias
		network.hiddenLayers[1].weights[0 .. $] = [ 0.1, 0.2, 0.3, 0.4,   // <- 1st neuron
		                                            0.5, 0.6, 0.7, 0.8,   // <- 2nd neuron
		                                            0.9, 1.0, 1.1, 1.2 ]; // <- 3rd neuron
		
		//                                                            bias
		network.hiddenLayers[2].weights[0 .. $] = [ -0.1, -0.2, -0.3, -0.4,   // <- 1st neuron
		                                            -0.5, -0.6, -0.7, -0.8,   // <- 2nd neuron
		                                            -0.9, -1.0, -1.1, -1.2 ]; // <- 3rd neuron
		
		network.outputLayer.weights[0 .. $] = [ 0, 1, 2, 3 ]; // the only neuron
		
		network(inputs, outputs, cublasHandle);
		cudaDeviceSynchronize();
		
		immutable float[] result = [4.497191, 4.500117, 4.501695, 4.502563];
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
	 * From more details look $(LINK std.math.kernels.cudaBLXa).
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
	void crossover(in Network x, in Network y, in float a, in float b, in float alpha, RandomPool pool) nothrow @nogc
	in
	{
		assert (this.depth == x.depth);
		assert (this.depth == y.depth);
		
		assert (a <= b);
		
		assert (alpha >= 0, "α parameter must be >= 0");
	}
	body
	{
		inputLayer.crossover (x.inputLayer,  y.inputLayer,  a, b, alpha, pool);
		outputLayer.crossover(x.outputLayer, y.outputLayer, a, b, alpha, pool);
		
		foreach (i, ref off; hiddenLayers)
			off.crossover(x.hiddenLayers[i], y.hiddenLayers[i], a, b, alpha, pool);
	}
	
	///
	unittest
	{
		mixin(writetest!crossover);
		
		import std.algorithm : max, min;
		import std.math      : abs;
		
		immutable NetworkParams params = {
			inputs  :  20,
			outputs :  10,
			neurons :  30,
			layers  :  10,
			min     : -1.0e3,
			max     :  1.0e3
		};
		
		immutable alpha = 0.5;
		
		auto parent1 = Network(params, randomPool);
		scope(exit) parent1.freeMem();
		
		auto parent2 = Network(params, randomPool);
		scope(exit) parent2.freeMem();
		
		auto offspring = Network(params, randomPool);
		scope(exit) offspring.freeMem();
		
		offspring.crossover(parent1, parent2, params.min, params.max, alpha, randomPool);
		cudaDeviceSynchronize();
		
		with (offspring)
		{
			with (inputLayer)
			{
				assert (weights.values.all!(x => isFinite(x)));
				
				foreach (i, w; weights)
				{
					float _min = min(parent1.inputLayer.weights[i], parent2.inputLayer.weights[i], params.min)
						- alpha * abs(parent1.inputLayer.weights[i] - parent2.inputLayer.weights[i]);
					
					float _max = max(parent1.inputLayer.weights[i], parent2.inputLayer.weights[i], params.max)
						+ alpha * abs(parent1.inputLayer.weights[i] - parent2.inputLayer.weights[i]);
					
					assert (w >= _min && w <= _max);
				}
			}
			
			with (outputLayer)
			{
				assert (weights.values.all!(x => isFinite(x)));
								
				foreach (i, w; weights)
				{
					float _min = min(parent1.outputLayer.weights[i], parent2.outputLayer.weights[i], params.min)
						- alpha * abs(parent1.outputLayer.weights[i] - parent2.outputLayer.weights[i]);
					
					float _max = max(parent1.outputLayer.weights[i], parent2.outputLayer.weights[i], params.max)
						+ alpha * abs(parent1.outputLayer.weights[i] - parent2.outputLayer.weights[i]);
					
					assert (w >= _min && w <= _max);
				}
			}
			
			assert (
				hiddenLayers.all!(
					l => l.weights.values.all!(
						x => isFinite(x)
			)));
			
			foreach (i, layer; hiddenLayers)
				foreach (j, w; layer.weights)
				{
					float _min = min(parent1.hiddenLayers[i].weights[j], parent2.hiddenLayers[i].weights[j], params.min)
						- alpha * abs(parent1.hiddenLayers[i].weights[j] - parent2.hiddenLayers[i].weights[j]);
					
					float _max = max(parent1.hiddenLayers[i].weights[j], parent2.hiddenLayers[i].weights[j], params.max)
						+ alpha * abs(parent1.hiddenLayers[i].weights[j] - parent2.hiddenLayers[i].weights[j]);
					
					assert (w >= _min && w <= _max);
				}
		}
	}
}

