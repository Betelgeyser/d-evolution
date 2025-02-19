/**
 * Copyright © 2017 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
import std.exception : enforce;
import std.json      : JSONValue;
import std.math      : isFinite;
import std.string    : format;

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
	import std.json      : parseJSON, toJSON;
	import std.math      : approxEqual;
	
	private RandomPool     randomPool;
	
	static this()
	{
		randomPool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, 0, 100_000);
	}
	
	static ~this()
	{
		randomPool.freeMem();
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

	float min = -1.0; /// Minimal generated weight.
	float max =  1.0; /// Maximal generated weight.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (neurons >= 1);
		assert (layers  >= 2); // There must be at least input and output layers
		
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
	private Layer[] _layers; /// Layers of the network.
	
	/**
	 * Returns: The total number of layers, including input and output layers.
	 */
	@property ulong depth() const @nogc nothrow pure @safe
	{
		return _layers.length;
	}
	
	/**
	 * Returns: The input (the first) layer.
	 */
	@property const(Layer) inputLayer() const @nogc nothrow
	{
		return _layers[0];
	}
	
	/**
	 * Returns: All hidden layers of the network.
	 */
	@property const(Layer[]) hiddenLayers() const @nogc nothrow pure @safe
	{
		return _layers[1 .. $-1];
	}
	
	/**
	 * Returns: The output (the last) layer.
	 */
	@property const(Layer) outputLayer() const @nogc nothrow
	{
		return _layers[$ - 1];
	}
	
	/**
	 * Returns: All network's layers.
	 */
	@property const(Layer[]) layers() const @nogc nothrow pure @safe
	{
		return _layers;
	}
	
	/**
	 * Returns: How many input values the network takes.
	 */
	@property uint inputs() const @nogc nothrow
	{
		return inputLayer.inputs;
	}
	
	/**
	 * Returns: How many output values the network returns.
	 */
	@property uint outputs() const @nogc nothrow
	{
		return outputLayer.neurons;
	}
	
	/**
	 * Returns: The number of neurons per a hidden layer.
	 */
	@property uint width() const @nogc nothrow
	{
		return inputLayer.neurons;
	}
	
	/**
	 * Constructs a random neural network acording to the given parameters.
	 *
	 * Params:
	 *     params = Network parameters.
	 *     pool = Pseudorandom number generator.
	 */
	this(in NetworkParams params, RandomPool pool)
	in
	{
		assert (&params, "Neural network parameters are incorrect.");
	}
	body
	{
		scope(failure) freeMem();
		
		_layers = nogcMalloc!Layer(params.layers);
		
		_layers[0]     = Layer(params.inputParams,  pool);
		_layers[$ - 1] = Layer(params.outputParams, pool);
		
		_layers[1 .. $-1].each!((ref x) => x = Layer(params.hiddenParams, pool));
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		auto pool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, 0, 1000);
		
		NetworkParams params = { layers : 4, inputs : 2, neurons : 3, outputs : 1 };
		
		Network network = Network(params, pool);
		scope(exit) network.freeMem();
		
		with (network)
		{
			assert (depth == params.layers);
			assert (width == params.neurons);
			
			assert (inputLayer.inputs  == params.inputs);
			assert (inputLayer.neurons == params.neurons);
			
			assert (outputLayer.inputs  == params.neurons);
			assert (outputLayer.neurons == params.outputs);
				
			assert (hiddenLayers.all!(l => l.inputs  == params.neurons));
			assert (hiddenLayers.all!(l => l.neurons == params.neurons));
			
			// Not that network should test layers, but need to check whether network creates all layers or not
			assert (
				_layers.all!(
					l => l.weights.all!(
						w => isFinite(w)))
			);
			assert (
				_layers.all!(
					l => l.weights.all!(
						w => w.between(params.min, params.max)))
			);
		}
	}
	
	this(in JSONValue json)
	{
		scope(failure) freeMem();
		
		auto JSONLayers = json["Layers"].array;
		
		_layers = nogcMalloc!Layer(JSONLayers.length);
		foreach(i, l; JSONLayers)
			_layers[i] = Layer(l);
	}
	
	unittest
	{
		mixin(writeTest!__ctor);
		
		immutable string str = `{
			"Inputs": 3,
			"Depth": 3,
			"Width": 2,
			"Outputs": 1,
			"Layers": [
				{
					"Inputs": 3,
					"Neurons": 2,
					"Weights": [0, 1, 2, 3, 4, 5, 6, 7]
				},
				{
					"Inputs": 2,
					"Neurons": 2,
					"Weights": [0, 1, 2, 3, 4, 5]
				},
				{
					"Inputs": 2,
					"Neurons": 1,
					"Weights": [0, 1, 2]
				}
			]
		}`;
		
		JSONValue json = parseJSON(str);
		
		auto network = Network(json);
		scope(exit) network.freeMem();
		
		with (network)
		{
			assert (inputs  == 3);
			assert (width   == 2);
			assert (depth   == 3);
			assert (outputs == 1);
		}
	}
	
	JSONValue json() const
	{
		JSONValue result = [
			"Inputs" : inputs,
			"Depth" : depth,
			"Outputs" : outputs,
			"Width" : width
		];
		
		JSONValue[] layersJSON;
		foreach(layer; layers)
			layersJSON ~= layer.json();
		
		result["Layers"] = layersJSON;
		
		return result;
	}
	
	unittest
	{
		mixin(writeTest!json);
		
		immutable string str = `{
			"Inputs": 3,
			"Depth": 3,
			"Width": 2,
			"Outputs": 1,
			"Layers": [
				{
					"Inputs": 3,
					"Neurons": 2,
					"Weights": [0, 1, 2, 3, 4, 5, 6, 7]
				},
				{
					"Inputs": 2,
					"Neurons": 2,
					"Weights": [0, 1, 2, 3, 4, 5]
				},
				{
					"Inputs": 2,
					"Neurons": 1,
					"Weights": [0, 1, 2]
				}
			]
		}`;
		
		JSONValue json = parseJSON(str);
		
		auto network = Network(json);
		scope(exit) network.freeMem();
		
		auto networkJSON = network.json;
		
		assert (
			networkJSON.toJSON ==
			`{"Depth":3,"Inputs":3,"Layers":[{"Inputs":3,"Neurons":2,"Weights":[0,1,2,3,4,5,6,7]},{"Inputs":2,"Neurons":2,"Weights":[0,1,2,3,4,5]},{"Inputs":2,"Neurons":1,"Weights":[0,1,2]}],"Outputs":1,"Width":2}`
		);
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
		_layers.each!(x => x.freeMem());
		
		if (_layers.length)
			nogcFree(_layers);
	}
	
	static void copy(in Network src, Network dst)
	{
		src._layers.each!((i, x) => Layer.copy(x, dst._layers[i]));
	}
	
	/**
	 * Activate the network.
	 *
	 * Claculates the result of feeding an input matrix to the network.
	 *
	 * Params:
	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
	 *     outputs = Output matrix of size m x n, where n is the number of output neurons.
	 *     cublasHandle = Cublas handle.
	 */
	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle) const
	{
		enforce(
			inputs.cols == this.inputs,
			"Inputs must have %d columns, got %d".format(this.inputs, inputs.cols)
		);
		
		auto inputsE = Matrix(inputs.rows, inputs.cols + biasLength); // Extended
		scope(exit) inputsE.freeMem();
		
		inputsE.colSlice(0, inputs.cols).values[0 .. $] = inputs.values[0 .. $];
		cudaFill(inputsE.colSlice(inputsE.cols - biasLength, inputsE.cols), biasWeight);
		
		auto prev = Matrix(inputs.rows, width + biasLength);
		scope(exit) prev.freeMem();
		
		auto next = Matrix(inputs.rows, width + biasLength);
		scope(exit) next.freeMem();
		
		cudaFill(prev.colSlice(prev.cols - biasLength, prev.cols), biasWeight);
		cudaFill(next.colSlice(next.cols - biasLength, next.cols), biasWeight);
		
		inputLayer()(inputsE, prev.colSlice(0, prev.cols - biasLength), cublasHandle);
		foreach (l; hiddenLayers)
		{
			l(prev, next.colSlice(0, next.cols - biasLength), cublasHandle);
			swap(prev, next);
		}
		outputLayer()(prev, outputs, cublasHandle, false);
	}
	
	///
	unittest
	{
//		mixin(writeTest!opCall);
//		
//		immutable NetworkParams params = { inputs : 2, outputs : 1, neurons : 3, layers : 5 };
//		
//		immutable measures = 4;
//		
//		auto inputs = Matrix(measures, params.inputs);
//		scope(exit) inputs.freeMem();
//		
//		auto outputs = Matrix(measures, params.outputs);
//		scope(exit) outputs.freeMem();
//		
//		auto network = Network(params, randomPool);
//		scope(exit) network.freeMem();
//		
//		inputs[0 .. $].each!"a = i + 1";
//		
//		// Reinitializing network with deterministic values for testing 
//		with (network)
//		{
//			_layers[0].weights.each!"a =  0.1 * (i + 1)";
//			_layers[1].weights.each!"a = -0.1 * (i + 1)";
//			_layers[2].weights.each!"a =  0.1 * (i + 1)";
//			_layers[3].weights.each!"a = -0.1 * (i + 1)";
//			_layers[4].weights.each!"a = i";
//		}
//		
//		network(inputs, outputs, cublasHandle);
//		cudaDeviceSynchronize();
//		
//		immutable float[] result = [4.497191, 4.500117, 4.501695, 4.502563];
//		assert (equal!approxEqual(outputs, result));
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
	void crossover(in Network x, in Network y, in float a, in float b, in float alpha, RandomPool pool)
	in
	{
		assert (this.depth == x.depth);
		assert (this.depth == y.depth);
		
		assert (a <= b);
		
		assert (alpha >= 0, "α parameter must be >= 0");
	}
	body
	{
		_layers.each!(
			(i, ref l) => l.crossover(x._layers[i], y._layers[i], a, b, alpha, pool)
		);
	}
	
	///
	unittest
	{
		mixin(writeTest!crossover);
		
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
		
		auto pool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, 0, 10000);
		
		immutable alpha = 0.5;
		
		auto parent1 = Network(params, pool);
		scope(exit) parent1.freeMem();
		
		auto parent2 = Network(params, pool);
		scope(exit) parent2.freeMem();
		
		auto offspring = Network(params, pool);
		scope(exit) offspring.freeMem();
		
		offspring.crossover(parent1, parent2, params.min, params.max, alpha, pool);
		cudaDeviceSynchronize();
		
		with (offspring)
		{
			assert (
				_layers.all!(
					l => l.weights.all!(
						x => isFinite(x)))
			);
			
			foreach (i, l; _layers)
				foreach (j, w; l.weights)
				{
					float diff = abs(parent1._layers[i].weights[j] - parent2._layers[i].weights[j]);
					
					float _min = min(parent1._layers[i].weights[j], parent2._layers[i].weights[j]);
					float _max = max(parent1._layers[i].weights[j], parent2._layers[i].weights[j]);
					
					_min -= alpha * diff;
					_max += alpha * diff;
					
					_min = max(_min, params.min);
					_max = min(_max, params.max);
					
					assert (w >= _min && w <= _max);
				}
		}
	}
}

