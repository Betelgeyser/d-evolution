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

version (unittest)
{
	import std.math : approxEqual;
	immutable accuracy = 0.000001;
}


immutable biasLength = 1; /// Number of bias weights per neuron.
immutable biasWeight = 1; /// Weight of every bias connection.

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
		assert (inputs  >= 1, "Neural network must have at least 1 input neuron.");
		assert (outputs >= 1, "Neural network must have at least 1 output neuron.");
		assert (neurons >= 1, "Neural network must have at least 1 neuron in every hidden layer.");
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
	Matrix weights; /// Connection weights.
	
	/**
	 * Number of connections per neuron (including bias).
	 */
	@property uint connections() const pure nothrow @safe @nogc
	{
		return weights.rows;
	}
	
	/**
	 * Number of neurons in the layer.
	 */
	@property uint neurons() const pure nothrow @safe @nogc
	{
		return weights.cols;
	}
	
	invariant
	{
		assert (&weights, "The weights matrix is incorrect.");
		assert (weights.rows >= 1 + biasLength, "A layer must have at least 2 connections."); // connections()
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
	in
	{
		assert (inputs  >= 1, "A layer must have at least 1 input connection.");
		assert (neurons >= 1, "A layer must have at least 1 neuron.");
	}
	body
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
		assert (inputs.cols == connections, "The number of matrix columns must be equal to the layer's connections number.");
		assert (inputs.rows == outputs.rows, "The input and the output matrix must have the same number of rows.");
		assert (neurons     <= outputs.cols, "The output matrix must not have less columns than the layer's neurons number.");
	}
	body
	{
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
		
		if (activate)
			cuda_tanh(outputs, outputs.rows * outputs.cols);
	}
	
	///
	unittest
	{
		mixin(writetest!opCall);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		scope(exit) curandDestroyGenerator(generator);
		
		// Initialize cuBLAS
		cublasHandle_t handle;
		cublasCreate(handle);
		scope(exit) cublasDestroy(handle);
		
		Layer l = Layer(2, 2, generator); scope(exit) l.freeMem();
		cudaDeviceSynchronize();
		
		/*   Neurons
		 *   V    V
		 * 0.00 0.06 * <- weights
		 * 0.02 0.08 * <- weights
		 * 0.04 0.10 * <- biases */
		for (ulong i = 0; i < l.weights.length; ++i)
			l.weights[i] = i / 50f;
		
		Matrix inputs;
		inputs.rows = 4;
		inputs.cols = 3;
		cudaMallocManaged(inputs, inputs.rows * inputs.cols);
		
		/* 0 4  8 *
		 * 1 5  9 *
		 * 2 6 10 *
		 * 3 7 11 */
		for (ulong i = 0; i < inputs.length; ++i)
			inputs[i] = i;
		
		Matrix outputs;
		outputs.rows = 4;
		outputs.cols = 2;
		cudaMallocManaged(outputs, outputs.rows * outputs.cols);
		
		l(inputs, outputs, handle);
		cudaDeviceSynchronize();
		
		/* 0.379949 0.807569 *
		 * 0.430084 0.876393 *
		 * 0.477700 0.921669 *
		 * 0.522665 0.950795 */
		immutable float[] result = [0.379949, 0.430084, 0.477700, 0.522665, 0.807569, 0.876393, 0.921669, 0.950795];
		for (ulong i = 0; i < outputs.length; ++i)
			assert (
				approxEqual(
					outputs[i], result[i],
					accuracy
				)
			);
	}
}

/**
 * Simple feedforward neural network.
 */
struct Network
{
	Layer  inputLayer;   /// Self explaining.
	Layer* hiddenLayers; /// ditto
	Layer  outputLayer;  /// ditto
	
	uint depth; /// Number of hidden layers (input and output does not count).
	
	invariant
	{
		for (ulong i = 0; i < depth; ++i)
			assert (
				hiddenLayers[i].neurons == inputLayer.neurons,
				"Every hidden layer must have the same number of neurons as the input layer."
			);
	}
	
	/**
	 * Get number of neurons per hidden layer.
	 */
	@property uint neuronsPerLayer() const pure nothrow @safe @nogc
	{
		return inputLayer.neurons;
	}
	
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
		assert (&params, "Neural network parameters are incorrect.");
	}
	body
	{
		scope(failure) freeMem();
		
		inputLayer  = Layer(params.inputs,  params.neurons, generator);
		outputLayer = Layer(params.neurons, params.outputs, generator);
		
		depth = params.layers;
		
		hiddenLayers = cast(Layer*)malloc(depth * Layer.sizeof);
		for (ulong i = 0; i < depth; ++i)
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
		
		assert (n.depth           == params.layers);
		assert (n.neuronsPerLayer == params.neurons);
		
		assert (n.inputLayer.connections == params.inputs + biasLength);
		assert (n.inputLayer.neurons     == params.neurons);
		
		assert (n.outputLayer.connections == params.neurons + biasLength);
		assert (n.outputLayer.neurons     == params.outputs);
		
		for (ulong i = 0; i < n.depth; ++i)
		{
			assert (n.hiddenLayers[i].connections == params.neurons + biasLength);
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
		
		if (depth > 0)
		{
			for (ulong i = 0; i < depth; ++i)
				hiddenLayers[i].freeMem();
			free(hiddenLayers);
		}
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
		auto prev = Matrix(inputs.cols, neurons + 1);
		auto next = Matrix(inputs.cols, neurons + 1);
		
		cuda_fill(prev + inputs.cols * neurons, 1, neurons + 1); // Extend matrix with 1's for biases
		inputLayer(inputs, prev, cublasHandle);
		
		for (int i = 0; i < depth; i++)
		{
			cuda_fill(prev + inputs.cols * neurons, 1, neurons + 1); // Reset last column to 1's
			hiddenLayers[i](prev, next, cublasHandle);
			
			auto tmp = prev;
			prev = next;
			next = prev;
		}
		
		cuda_fill(prev + inputs.cols * neurons, 1, neurons + 1); // Reset last column to 1's
		outputLayer(prev, outputs, cublasHandle, false);
	}
	
	///
	unittest
	{
		mixin(writetest!opCall);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		scope(exit) curandDestroyGenerator(generator);
		
		// Initialize cuBLAS
		cublasHandle_t handle;
		cublasCreate(handle);
		scope(exit) cublasDestroy(handle);
		
		NetworkParams params;
		params.inputs  = 1;
		params.outputs = 1;
		params.neurons = 1;
		params.layers  = 1;
		
		/* 0 1 *
		 * 1 1 */
		auto inputs = Matrix(2, 2);
		cudaDeviceSynchronize();
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
		network.inputLayer.weights[0]   =  0.00;
		network.inputLayer.weights[1]   =  1.00; // bias
		network.hiddenLayers.weights[0] =  2.00;
		network.hiddenLayers.weights[1] = -2.00; // bias
		network.outputLayer.weights[0]  = -0.50;
		network.outputLayer.weights[1]  =  0.75; // bias
		
		network(inputs, outputs, handle);
		cudaDeviceSynchronize();
		
		float[] result = [0.971843, 0.971843];
		for(ulong i = 0; i < outputs.length; ++i)
			assert (
				approxEqual(
					outputs[i], result[i],
					accuracy
				)
			);
	}
}

