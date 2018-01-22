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

// C libs
import core.stdc.stdlib;
import std.exception;
import std.conv;
import std.traits;
import std.random : unpredictableSeed;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import neural.layer;

void fun(float* x)
{
	for (int i = 0; i < 100; i++)
//		x[i] += y[i];
		x[i] = i;
}

/**
 * Simple feedforward network.
 */
struct Network
{
	Layer inputLayer;
	Layer hiddenLayer;
	Layer outputLayer;
	
	static void randomPopulation(ref Network* population, in NetworkParams params, in uint size)
	in
	{
		assert (&params);
	}
	body
	{
		import std.stdio;
		curandStatus_t curandStatus;
		
		curandGenerator_t generator;
		enforceCurand(curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT));
		enforceCurand(curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed()));
		
		scope(exit) enforceCurand(curandDestroyGenerator(generator));
		
		population = cast(Network*)malloc(size * Network.sizeof);
		scope(failure) free(population);
		
		for (int i = 0; i < size; i++)
		{
			population[i].inputLayer  = Layer(params.inputs,  params.neurons);
			population[i].hiddenLayer = Layer(params.neurons, params.neurons);
			population[i].outputLayer = Layer(params.neurons, params.outputs);
			
			enforceCurand(curandGenerate(generator, population[i].inputLayer,  params.inputs  * params.neurons));
			enforceCurand(curandGenerate(generator, population[i].hiddenLayer, params.neurons * params.neurons));
			enforceCurand(curandGenerate(generator, population[i].outputLayer, params.outputs * params.neurons));
		}
	}
	
	unittest
	{
		import std.stdio;
		
		writeln("Network.randomPopulation(ref Network* population, in NetworkParams params, in uint size)");
		
		NetworkParams params;
		params.inputs  = 1;
		params.neurons = 2;
		params.outputs = 1;
		
		uint number = 1;
		
		Network* population;
		scope(exit) free(population);
		
		writeln(">>> Generating random population of ", number, " networks with parameters: ", params);
		randomPopulation(population, params, number);
		
		float* input;  scope(exit) free(input);
		float* hidden; scope(exit) free(hidden);
		float* output; scope(exit) free(output);
		
		with(population[0])
		{
			input  = cast(float*)malloc(inputLayer.size);
			hidden = cast(float*)malloc(hiddenLayer.size);
			output = cast(float*)malloc(outputLayer.size);
			
			cudaMemcpy(input,  inputLayer,  inputLayer.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
			cudaMemcpy(hidden, hiddenLayer, hiddenLayer.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
			cudaMemcpy(output, outputLayer, outputLayer.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		
			write(">>> Resulting input  layer = [");
			for (uint i = 0; i < inputLayer.length; i++)
				if (i != inputLayer.length - 1)
					writef("% e, ", input[i]);
				else
					writefln("% e]", input[i]);
			
			write(">>> Resulting hidden layer = [");
			for (uint i = 0; i < hiddenLayer.length; i++)
				if (i != hiddenLayer.length - 1)
					writef("% e, ", hidden[i]);
				else
					writefln("% e]", hidden[i]);
			
			write(">>> Resulting output layer = [");
			for (uint i = 0; i < outputLayer.length; i++)
				if (i != outputLayer.length - 1)
					writef("% e, ", output[i]);
				else
					writefln("% e]", output[i]);
		}
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

struct Layer
{
	alias weights this;
	
	float* weights;
	int    weightsPerNeuron;
	int    neuronsNum;
	
	/**
	 * Default constructor.
	 *
	 * Params:
	 *     length = Number of neurons.
	 */
	this(int weightsPerNeuron, int neuronsNum) nothrow @nogc
	{
		this.weightsPerNeuron = weightsPerNeuron;
		this.neuronsNum       = neuronsNum;
		cudaMalloc(weights, this.length);
	}
	
	~this() { cudaFree(weights); }
	
	@property length()
	{
		return weightsPerNeuron * neuronsNum;
	}
	
	@property size()
	{
		return weightsPerNeuron * neuronsNum * float.sizeof;
	}
	
	unittest
	{
		import std.stdio;
		import core.stdc.stdlib;
		
		writeln("Layer.this(float* weights, int weightsPerNeuron, int neuronsNum) pure nothrow @safe @nogc");
		
		auto l = Layer(1, 1);
	}
}

/**
 * Random network generation parameters.
 */
struct NetworkParams
{
	private struct Weights
	{
		float min = -float.max;
		float max =  float.max;
		
		invariant { assert (min <= max); }
		
		@property string toString()
		{
			return "Weights(min = " ~ min.to!string ~ ", max = " ~ max.to!string ~ ")";
		}
	}
	uint    inputs;  /// Number of network's inputs.
	uint    outputs; /// Number of network's outputs.
	uint    neurons; /// Number of neurons in every hidden layer.
	Weights weights; /// Allowed neuron weights bounds.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (neurons >= 1);
		
		assert(&weights);
	}
	
	@property string toString()
	{
		return "NetworkParams(inputs = " ~ inputs.to!string ~ ", outputs = " ~ outputs.to!string ~ ", neurons = " ~ neurons.to!string ~ ", weights = " ~ weights.to!string ~ ")";
	}
}

