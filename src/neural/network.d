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
import std.random : unpredictableSeed;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

/**
 * Simple feedforward network.
 *
 * Currently supports only two layers, excluding output layer.
 */
struct Network
{
	Layer inputLayer;  /// First of the two layers.
	Layer hiddenLayer; /// Second of the two layers.
	Layer outputLayer; /// Output layer.
	                   /// The only difference with previous two is that activation function does not appy to the output layer.
	
	/**
	 * Generate random population.
	 *
	 * Params:
	 *     params = Parameters for network generation.
	 *     size = Number of individuals in a population.
	 *
	 * Returns:
	 *     Reference to a pointer to a population array.
	 */
	static void randomPopulation(ref Network* population, in NetworkParams params, in uint size)
	in
	{
		assert (&params);
	}
	body
	{
		population = cast(Network*)malloc(size * Network.sizeof);
		
		curandGenerator_t generator;
		enforceCurand(curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT));
		enforceCurand(curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed()));
		
		scope(exit) enforceCurand(curandDestroyGenerator(generator));
		
		population = cast(Network*)malloc(size * Network.sizeof);
		scope(failure) free(population);
		
		for (int i = 0; i < size; i++)
			with (population[i])
			{
				inputLayer  = Layer(params.inputs,  params.neurons);
				hiddenLayer = Layer(params.neurons, params.neurons);
				outputLayer = Layer(params.neurons, params.outputs);
				
				enforceCurand(curandGenerate(generator, inputLayer,  inputLayer.length));
				enforceCurand(curandGenerate(generator, hiddenLayer, hiddenLayer.length));
				enforceCurand(curandGenerate(generator, outputLayer, outputLayer.length));
			}
	}
	
	unittest
	{
		import std.stdio;
		
		writeln("Network.randomPopulation(ref Network* population, in NetworkParams params, in uint size)");
		
		NetworkParams params;
		params.inputs  = 2;
		params.neurons = 3;
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
	
	static immutable biasLength = 1; /// Number of bias weights per neuron.
	
	float* weights;          /// Array of neurons weights.
	uint   weightsPerNeuron; /// Number of weights of each neuron.
	uint   neuronsNum;       /// Number of neurons in the layer.
	
	/**
	 * Default constructor.
	 *
	 * Params:
	 *     weightsPerNeuron = Number of weights per neuron.
	 *     neuronsNum = Number of neurons in the layer.
	 */
	this(uint weightsPerNeuron, uint neuronsNum) nothrow @nogc
	{
		this.weightsPerNeuron = weightsPerNeuron;
		this.neuronsNum       = neuronsNum;
		cudaMalloc(weights, this.length);
	}
	
	/**
	 * Free memory.
	 */
	~this()
	{
		cudaFree(weights);
	}
	
	/**
	 * Number of elements in the weights array.
	 */
	@property length()
	{
		return (weightsPerNeuron + biasLength) * neuronsNum;
	}
	
	/**
	 * Size of the weights array in bytes.
	 */
	@property size()
	{
		return this.length * float.sizeof;
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
	uint    inputs;  /// Number of network's inputs.
	uint    outputs; /// Number of network's outputs.
	uint    neurons; /// Number of neurons in every hidden layer.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (neurons >= 1);
	}
	
	/**
	 * String representation.
	 */
	@property string toString()
	{
		return "NetworkParams(inputs = " ~ inputs.to!string ~ ", outputs = " ~ outputs.to!string ~ ", neurons = " ~ neurons.to!string ~ ")";
	}
}

