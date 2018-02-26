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
 */
module neural.network;

// C modules
import core.stdc.stdlib;

// D modules
import std.format;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import common;
import math;


/**
 * Random network generation parameters.
 */
struct NetworkParams
{
	uint inputs;  /// Number of network's inputs.
	uint outputs; /// Number of network's outputs.
	uint layers;  /// Number of hidden layers (excluding input and output layers).
	uint neurons; /// Number of neurons in every hidden layer.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (neurons >= 1);
		assert (layers  >= 0);
	}
	
	/**
	 * String representation.
	 */
	@property string toString() pure const @safe
	{
		return "NetworkParams(inputs = %d, outputs = %d, neurons = %d)".format(
			inputs,
			outputs,
			neurons
		);
	}
}

/**
 * Feedforward layer.
 *
 * Each neuron of this layer is connected to each neuron of the previous layer.
 */
struct Layer
{
	static immutable uint biasLength = 1; /// Number of bias weights per neuron.
	
	Matrix weights; /// Connection weights.
	
	/**
	 * Number of connections per neuron (including bias).
	 */
	@property void connections(in uint val) pure nothrow @safe @nogc
	{
		weights.rows = val;
	}
	
	/// ditto
	@property uint connections() const pure nothrow @safe @nogc
	{
		return weights.rows;
	}
	
	/**
	 * Number of neurons in the layer.
	 */
	@property void neurons(in uint val) pure nothrow @safe @nogc
	{
		weights.cols = val;
	}
	
	/// ditto
	@property uint neurons() const pure nothrow @safe @nogc
	{
		return weights.cols;
	}
	
	invariant
	{
		assert (&weights);
		assert (weights.rows >= 1 + biasLength); // connections()
	}
	
	/**
	 * Constructor for random layer.
	 *
	 * Params:
	 *     inputs = Number of weights per neuron.
	 *     neurons = Number of neurons in the layer.
	 *     generator = Pseudorandom number generator.
	 */
	this(in uint inputs, in uint neurons, curandGenerator_t generator) nothrow @nogc
	{
		scope(failure) freeMem();
		
		weights = Matrix(
			inputs + biasLength,
			neurons,
			generator
		);
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		
		scope(exit) curandDestroyGenerator(generator);
		
		auto l = Layer(3, 2, generator); scope(exit) l.freeMem();
		cudaDeviceSynchronize();
		
		assert (l.connections == 3 + biasLength);
		assert (l.neurons     == 2);
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
	 * column to multiply bias on.
	 *
	 * Params:
	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
	 *     outputs = Output matrix of size m x n, where n is the number of neurons.
	 *     cublasHandle = Cublas handle.
	 */
	void opCall(in Matrix inputs, Matrix outputs) const nothrow @nogc
	{
		cublasHandle_t handle;
		cublasCreate(handle);
		scope(exit) cublasDestroy(handle);
		
		opCall(inputs, outputs, handle);
	}
	
	/// ditto
	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle) const nothrow @nogc
	{
		assert (inputs.cols == connections);
		assert (inputs.rows == outputs.rows);
		assert (neurons     <= outputs.cols);
		
		immutable float alpha = 1;
		immutable float beta  = 0;
		
		cublasSgemm(
			cublasHandle,
			cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
			inputs.rows, neurons, connections,
			&alpha,
			inputs, inputs.rows,
			weights, connections,
			&beta,
			outputs, inputs.rows
		);
		
		cuda_tanh(outputs, outputs.rows * outputs.cols);
	}
	
	///
	unittest
	{
		import std.math : approxEqual;
		mixin(writetest!opCall);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		
		scope(exit) curandDestroyGenerator(generator);
		
		Layer l = Layer(2, 2, generator); scope(exit) l.freeMem();
		cudaDeviceSynchronize();
		
		/* 0.00 0.08 0.16 *
		 * 0.02 0.10 0.18 *
		 * 0.04 0.12 0.20 *
		 * 0.06 0.14 0.22 */
		for (int i = 0; i < l.weights.length; i++)
			l.weights[i] = i / 50f;
		
		Matrix inputs;
		inputs.rows = 4;
		inputs.cols = 3;
		cudaMallocManaged(inputs, inputs.rows * inputs.cols);
		
		/* 0 3 *
		 * 1 4 *
		 * 2 5 */
		for (int i = 0; i < inputs.rows * inputs.cols; i++)
			inputs[i] = i;
		
		Matrix outputs;
		outputs.rows = 4;
		outputs.cols = 2;
		cudaMallocManaged(outputs, outputs.rows * outputs.cols);
		
		l(inputs, outputs);
		cudaDeviceSynchronize();
		
		/* 0.379949 0.807569 *
		 * 0.430084 0.876393 *
		 * 0.477700 0.921669 *
		 * 0.522665 0.950795 */
		immutable float[] result = [0.379949, 0.430084, 0.477700, 0.522665, 0.807569, 0.876393, 0.921669, 0.950795];
		for (int i = 0; i < outputs.rows * outputs.cols; i++)
			assert (
				approxEqual(
					outputs[i], result[i],
					0.0001
				)
			);
	}
}

/**
 * Simple feedforward neural network.
 */
struct Network
{
//	static const(Data)* trainingData;
	static cublasHandle_t cublasHandle;
	
	Layer  inputLayer;   /// Self explaining.
	Layer* hiddenLayers; /// ditto
	Layer  outputLayer;  /// ditto
	
	uint depth;   /// Number of hidden layers (input and output does not count).
	uint neurons; /// Number of neurons per layer (except the outputLayer).
	
	/**
	 * Constructor for random neural network.
	 *
	 * Params:
	 *     params = Network parameters.
	 *     generator = Pseudorandom number generator.
	 */
	this(in NetworkParams params, curandGenerator_t generator) nothrow @nogc
	in
	{
		assert (&params);
	}
	body
	{
		scope(failure) freeMem();
				
		inputLayer  = Layer(params.inputs,  params.neurons, generator);
		outputLayer = Layer(params.neurons, params.outputs, generator);
		
		depth   = params.layers;
		neurons = params.neurons;
		hiddenLayers = cast(Layer*)malloc(depth * Layer.sizeof);
		for (uint i = 0; i < depth; i++)
			hiddenLayers[i] = Layer(params.neurons, params.neurons, generator);
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
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		
		scope(exit) curandDestroyGenerator(generator);
		
		Network n = Network(params, generator); scope(exit) n.freeMem();
		cudaDeviceSynchronize();
		
		assert (n.depth   == params.layers);
		assert (n.neurons == params.neurons);
		
		assert (n.inputLayer.connections  == params.inputs + 1);
		assert (n.inputLayer.neurons      == params.neurons);
		
		assert (n.outputLayer.connections == params.neurons + 1);
		assert (n.outputLayer.neurons     == params.outputs);
		
		for (int i = 0; i < n.depth; i++)
		{
			assert (n.hiddenLayers[i].connections == params.neurons + 1);
			assert (n.hiddenLayers[i].neurons     == params.neurons);
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
		
		if (hiddenLayers !is null)
		{
			for (uint i = 0; i < depth; i++)
				hiddenLayers[i].freeMem();
			free(hiddenLayers);
		}
	}
	
//	/**
//	 * Evaluate the layer.
//	 *
//	 * Evaluates a result of feeding inpit matrix to the network.
//	 *
//	 * Params:
//	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
//	 *     outputs = Output matrix of size m x n, where n is the number of output neurons.
//	 *     cublasHandle = Cublas handle.
//	 */
//	void opCall(in Matrix inputs, Matrix outputs)
//	{
//		cublasHandle_t handle;
//		cublasCreate(handle);
//		scope(exit) cublasDestroy(handle);
//		
//		opCall(inputs, outputs, handle);
//	}
//	
//	/// ditto
//	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle)
//	{
//		Matrix prev = Matrix(inputs.cols, neurons);
//		Matrix next = Matrix(inputs.cols, neurons);
//		
//		inputLayer(inputs, prev, cublasHandle);
//		
//		for (int i = 0; i < depth; i++)
//		{
//			hiddenLayers[i](prev, next, cublasHandle);
//			prev.freeMem();
//			prev = next;
//			next = Matrix(inputs.cols, neurons);
//		}
//		
//		outputLayer(prev, outputs, cublasHandle);
//	}
	
	///
	unittest
	{
//		mixin(writetest!opCall);
//		import std.stdio : writeln;
//		writeln("Network");
//		Genome g;
//		
//		g.input = 2;
//		
//		g.hidden = [
//			[ [1, 2, 3   ], [3, 2, 1   ], [1, 0, 1] ],
//			[ [1, 1, 1, 1], [2, 2, 2, 2]            ],
//			[ [2, 1, 2   ]                          ]
//		];
//		
//		Network n = Network(g);
//		assert (n.length             == 3);
//		assert (n.inputLayer.length  == 2);
//		assert (n.outputLayer.length == 1);
//		
//		n([0, 0]);
	}
}

