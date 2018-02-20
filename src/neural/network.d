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
import std.conv : to;

version (unittest)
{
	import std.stdio;
	import std.random : unpredictableSeed;
}

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import common;
import math;


extern(C++) void kernel_tanh(float* x);//, int n);// nothrow @nogc;


/**
 * Random network generation parameters.
 */
struct NetworkParams
{
	ushort inputs;  /// Number of network's inputs.
	ushort outputs; /// Number of network's outputs.
	ushort layers;  /// Number of hidden layers (excluding input and output layers).
	ushort neurons; /// Number of neurons in every hidden layer.
	
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

struct Layer
{
	static immutable ushort biasLength = 1; /// Number of bias weights per neuron.
	
	Matrix weights; /// Connections' weights of every neuron.
	
	/// Number of connections per neuron (including bias).
	@property void connections(in ushort val) pure nothrow @safe @nogc
	{
		weights.rows = val;
	}
	
	/// Number of neurons in the layer.
	@property void neurons(in ushort val) pure nothrow @safe @nogc
	{
		weights.cols = val;
	}
	
	/// Number of connections per neuron (including bias).
	@property ushort connections() const pure nothrow @safe @nogc
	{
		return weights.rows;
	}
	
	/// Number of neurons in the layer.
	@property ushort neurons() const pure nothrow @safe @nogc
	{
		return weights.cols;
	}
	
	/**
	 * Random layer.
	 *
	 * Params:
	 *     inputs = Number of weights per neuron.
	 *     neurons = Number of neurons in the layer.
	 *     generator = Pseudorandom number generator.
	 */
	this(in ushort inputs, in ushort neurons, curandGenerator_t generator) nothrow @nogc
	in
	{
		assert (inputs  >= 1);
		assert (neurons >= 1);
	}
	body
	{
		scope(failure) freeMem();
		
		this.connections = cast(ushort)(inputs + biasLength); // WTF? Error: cannot implicitly convert expression (cast(int)inputs + 1) of type int to ushort
		this.neurons     = neurons;
		
		cudaMallocManaged(weights.values, length);
		
		curandGenerate(generator, weights.values, length);
	}
	
	unittest
	{
		mixin(writetest!__ctor);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
		
		scope(exit) curandDestroyGenerator(generator);
		
		Layer l = Layer(3, 2, generator); scope(exit) l.freeMem();
		cudaDeviceSynchronize();
		
		assert (l.connections == 3 + biasLength);
		assert (l.neurons     == 2);
		assert (l.length      == 8);
		assert (l.size        == 32);
		
		assert (l.weights[0           ] == l.weights[0           ]);
		assert (l.weights[l.length - 1] == l.weights[l.length - 1]);
	}
	
	/**
	 * Free memory.
	 */
	void freeMem() nothrow @nogc
	{
//		cudaFree(weights);
	}
	
	void opCall(in Matrix inputs, Matrix outputs)// const nothrow @nogc
	{
		cublasHandle_t handle;
		cublasCreate(handle);
		scope(exit) cublasDestroy(handle);
		
		opCall(inputs, outputs, handle);
	}
	
	void opCall(in Matrix inputs, Matrix outputs, cublasHandle_t cublasHandle)// const nothrow @nogc
	{
		assert (inputs.cols == connections);
		assert (inputs.rows == outputs.rows);
		assert (neurons == outputs.cols);
		
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
		cudaDeviceSynchronize();
		
//		int* size;
//		cudaMallocManaged(size, 1);
//		size[0] = outputs.rows * outputs.cols;
		void** args;
		cudaMallocManaged!(void*)(args, 1);
		args[0] = outputs.values;
//		args[1] = size;
		cudaLaunchKernel(
			&kernel_tanh,
			dim3(1, 1, 1),
			dim3(1, 1, 1),
			args
		);
		cudaDeviceSynchronize();
		for (int i = 0; i < 8; i++)//size[0]; i++)
			writeln(outputs[i]);
//		cudaDeviceSynchronize();
	}
	
	unittest
	{
		mixin(writetest!opCall);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
		
		scope(exit) curandDestroyGenerator(generator);
		
		Layer l = Layer(2, 2, generator); scope(exit) l.freeMem();
		cudaDeviceSynchronize();
		
		for (int i = 0; i < l.length; i++)
			l.weights.values[i] = i;
		
		Matrix inputs;
		inputs.rows = 4;
		inputs.cols = 3;
		cudaMallocManaged(inputs, inputs.rows * inputs.cols);
		for (int i = 0; i < inputs.rows * inputs.cols; i++)
			inputs[i] = i;
		
		Matrix outputs;
		outputs.rows = 4;
		outputs.cols = 2;
		cudaMallocManaged(outputs, outputs.rows * outputs.cols);
		
		l(inputs, outputs);
//		cudaDeviceSynchronize();
		
//		immutable float[] result = [20, 23, 26, 29, 56, 68, 80, 92];
//		for (int i = 0; i < outputs.rows * outputs.cols; i++)
////			assert (outputs[i] == result[i]);
//			writeln(outputs[i]);
//			writeln(__LINE__);
	}
	
	/**
	 * Number of elements in the weights array.
	 */
	@property ushort length() pure const nothrow @nogc
	{
		return cast(ushort) (connections * neurons);
	}
	
	/**
	 * Size of the weights array in bytes.
	 */
	@property ulong size() pure const nothrow @nogc
	{
		return length * float.sizeof;
	}
}

/**
 * Simple feedforward network.
 *
 * Currently supports only two layers, excluding output layer.
 */
struct Network
{
//	static const(Data)* trainingData;
	static cublasHandle_t cublasHandle;
	
	Layer  inputLayer;   /// Self explaining.
	Layer* hiddenLayers; /// Ditto.
	Layer  outputLayer;  /// Ditto.
	
	uint depth; /// Number of hidden layers (input and output does not count).
	
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
		
		depth = params.layers;
		hiddenLayers = cast(Layer*)malloc(depth * Layer.sizeof);
		for (uint i = 0; i < depth; i++)
			hiddenLayers[i] = Layer(params.neurons, params.neurons, generator);
	}
		
	unittest
	{
//		mixin(writetest!__ctor);
//		
//		NetworkParams params;
//		params.inputs  = 2;
//		params.layers  = 2;
//		params.neurons = 3;
//		params.outputs = 1;
//		
//		// Initialize cuRAND generator.
//		curandGenerator_t generator;
//		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
//		curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
//		
//		scope(exit) curandDestroyGenerator(generator);
//		
//		Network n = Network(params, generator); scope(exit) n.freeMem();
//		cudaDeviceSynchronize();
//		
//		assert (n.depth == params.layers);
//		
//		assert (n.inputLayer.length  == (params.inputs  + 1) * params.neurons);
//		assert (n.outputLayer.length == (params.neurons + 1) * params.outputs);
//		
//		// Check memory
//		assert (n.hiddenLayers[0].length == (params.neurons + 1) * params.neurons);
//		assert (n.hiddenLayers[params.layers - 1].length == (params.neurons + 1) * params.neurons);
	}
	
	/**
	 * Free memory.
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
	
	void evaluate() const nothrow @nogc
	{
//		//		int lda=m,ldb=k,ldc=m;
//		assert (trainingData.vectorLength == inputLayer.weightsPerNeuron);
//		
//		const float alpha = 1;
//		const float beta  = 0;
//		
//		float* layerResult;
//		layerResult = 
//		
//		cublasSgemm(
//			cublasHandle,
//			cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
//			trainingData.dataLength, trainingData.vectorLength, inputLayer.neuronsNum,
//			&alpha,
//			trainingData, trainingData.dataLength,
//			inputLayer, inputLayer.weightsPerNeuron,
//			&beta,
//			C, ldc);
	//	
//	// Destroy the handle
//	cublasDestroy(handle);
//		cublasSgemm(handle,
//		cublasOperation_t transa,
//		cublasOperation_t transb,
//		int m,
//		int n,
//		int k,
//		const float *alpha,
//		const float *A,
//		int lda,
//		const float *B,
//		int ldb,
//		const float *beta,
//		float *C,
//		int ldc
//	);
	}
	
	/**
	 *
	 *
	 * Params:
	 *     input = Input values to work on.
	 */
	void opCall(immutable float* input)
	{
//		int lda=m,ldb=k,ldc=m;
//	const float alf = 1;
//	const float bet = 0;
//	const float *alpha = &alf;
//	const float *beta = &bet;
//	
//	// Create a handle for CUBLAS
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//	
//	// Do the actual multiplication
//	cublasSgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//	
//	// Destroy the handle
//	cublasDestroy(handle);
//		cublasSgemm(handle,
//		cublasOperation_t transa,
//		cublasOperation_t transb,
//		int m,
//		int n,
//		int k,
//		const float *alpha,
//		const float *A,
//		int lda,
//		const float *B,
//		int ldb,
//		const float *beta,
//		float *C,
//		int ldc
//	);
	}
//		inputLayer(input);
//		
//		foreach (i, ref h; hiddenLayers)
//			if (i == 0)
//				h(inputLayer);
//			else
//				h(hiddenLayers[i - 1]);
//		
//		return hiddenLayers[$ - 1]();
//	}
	
//	unittest
//	{
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
//	}
//	
//	/**
//	 * Return hidden layers number.
//	 */
//	@property size_t length() const pure nothrow @safe @nogc
//	{
//		return hiddenLayers.length;
//	}
//	
//	/**
//	 * Human-readable string representation.
//	 */
//	@property string toString() const @safe
//	{
//		string result = "Network:\n";
//		result ~= inputLayer.toString("\t");
//		foreach(i, h; hiddenLayers)
//			result ~= h.toString("\t", i);
//		return result;
//	}
}

