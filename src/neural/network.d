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

struct Data
{
	alias data this;
	
	float* data;         /// Pointer to the data array itself.
	uint   vectorLength; /// Size of an input vector.
	uint   dataLength;   /// Number of measurements.
}

struct Layer
{
	static immutable ushort biasLength = 1; /// Number of bias weights per neuron.
	
	float*  weights;     /// Array of neurons weights.
	ushort* connections; /// Number of weights per neuron.
	ushort* neurons;     /// Number of neurons in the layer.
	
	/**
	 * Random layer.
	 *
	 * Params:
	 *     inputs = Number of weights per neuron.
	 *     neurons = Number of neurons in the layer.
	 *     generator = Pseudorandom number generator.
	 */
	this(in ushort inputs, in ushort neurons, ref curandGenerator_t generator) nothrow @nogc
	in
	{
		assert (inputs  >= 1);
		assert (neurons >= 1);
	}
	body
	{
		scope(failure) freeMem();
		
		cudaMallocManaged(this.connections, 1);
		cudaMallocManaged(this.neurons,     1);
		
		*this.connections = cast(ushort)(inputs + biasLength); // WTF? Error: cannot implicitly convert expression (cast(int)inputs + 1) of type int to ushort
		*this.neurons     = neurons;
		
		cudaMallocManaged(this.weights, length);
		
		curandGenerate(generator, weights, length);
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
		
		
		assert (*l.connections == 3 + biasLength);
		assert (*l.neurons     == 2);
		assert ( l.length      == 8);
		assert ( l.size        == 32);
		
		assert (l.weights[0           ] == l.weights[0           ]);
		assert (l.weights[l.length - 1] == l.weights[l.length - 1]);
	}
	
	/**
	 * Free memory.
	 */
	void freeMem() nothrow @nogc
	{
		if (weights !is null)
			cudaFree(weights);
		if (connections !is null)
			cudaFree(connections);
		if (neurons !is null)
			cudaFree(neurons);
	}
	
	void opCall(in float* data, ref cublasHandle_t cublasHandle) const nothrow @nogc
	{
//		int N = cols * rows;
//	
//	float* x_h;
//	float* y_h;
//	float* z_h;
//	
//	float* x_d;
//	float* y_d;
//	float* z_d;
//	
//	dudaMalloc(x_d, N);
//	dudaMalloc(y_d, N);
//	dudaMalloc(z_d, N);
//	x_h = cast(float*)malloc(N * float.sizeof);
//	y_h = cast(float*)malloc(N * float.sizeof);
//	z_h = cast(float*)malloc(N * float.sizeof);
//	
//	for (int i = 0; i < cols; i++)
//	{
//		for (int j = 0; j < rows; j++)
//		{
//			x_h[j + rows * i] = i + j;
//			y_h[j + rows * i] = i + j;
////			writeln("i = ", i, "; j = ", j, "; x_h[", j + rows * i, "] = ", x_h[j + rows * i]);
//		}
//	}
//	
//	cudaMemcpy(x_d, x_h, N * float.sizeof, cudaMemcpyKind.cudaMemcpyHostToDevice);
//	cudaMemcpy(y_d, y_h, N * float.sizeof, cudaMemcpyKind.cudaMemcpyHostToDevice);
//	
//	gpu_blas_mmul(x_d, y_d, z_d, 3, 3, 3);
//	
//	cudaMemcpy(z_h, z_d, N * float.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//	
////	for (int i = 0; i < N; i++)
////		writeln(z_h[i]);
//	
//	// Free memory
//	scope(exit)
//	{
//		cudaFree(x_d);
//		cudaFree(y_d);
//		cudaFree(z_d);
//		free(x_h);
//		free(y_h);
//		free(z_h);
//	}
	}
	
	/**
	 * Number of elements in the weights array.
	 */
	@property ushort length() pure const nothrow @nogc
	{
		return cast(ushort) (*connections * *neurons);
	}
	
	@property ushort opDollar() pure const nothrow @nogc
	{
		return length;
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
	static const(Data)* trainingData;
	static cublasHandle_t cublasHandle;
	
	Layer  inputLayer;   /// Self explaining.
	Layer* hiddenLayers; /// Ditto.
	Layer  outputLayer;  /// Ditto.
	
	uint depth; /// Number of hidden layers (input and output does not count).
	
	this(in NetworkParams params, ref curandGenerator_t generator)// nothrow @nogc
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
		mixin(writetest!__ctor);
		
		NetworkParams params;
		params.inputs  = 2;
		params.layers  = 2;
		params.neurons = 3;
		params.outputs = 1;
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
		
		scope(exit) curandDestroyGenerator(generator);
		
		Network n = Network(params, generator); scope(exit) n.freeMem();
		
		assert (n.depth == params.layers);
		
		assert (n.inputLayer.length  == (params.inputs  + 1) * params.neurons);
		assert (n.outputLayer.length == (params.neurons + 1) * params.outputs);
		
		// Check memory
		assert (n.hiddenLayers[0].length == (params.neurons + 1) * params.neurons);
		assert (n.hiddenLayers[params.layers - 1].length == (params.neurons + 1) * params.neurons);
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

