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
import std.conv   : to;
import std.random : unpredictableSeed;
import std.format;

debug import std.stdio;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

//import std.traits;

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
	
	uint hiddenNumber; /// Number of hidden layers (input and output does not count).
	
	void freeMem() nothrow @nogc
	{
		inputLayer.freeMem();
		outputLayer.freeMem();
		
		if (hiddenLayers !is null)
		{
			for (uint i = 0; i < hiddenNumber; i++)
				hiddenLayers[i].freeMem();
		
			free(hiddenLayers);
		}
	}
	
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
	static void randomPopulation(ref Network* population, in NetworkParams params, in uint size, in Data* trainingData = null)
	in
	{
		assert (&params);
	}
	body
	{
		Network.trainingData = trainingData;
		
		// Create a handle for cuBLAS
//		cublasCreate(Network.cublasHandle);
		
		// Initialize cuRAND generator.
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
		
		scope(exit) curandDestroyGenerator(generator);
		
		// Allocate memory for population.
		population = cast(Network*)malloc(size * Network.sizeof);
		scope(failure)
		{
			for (int i = 0; i < size; i++)
				population[i].freeMem();
			free(population);
		}
		
		for (int i = 0; i < size; i++)
			with (population[i])
			{
				hiddenNumber = params.layers;
				
				inputLayer  = Layer(params.inputs,  params.neurons, generator);
				outputLayer = Layer(params.neurons, params.outputs, generator);
				
				hiddenLayers = cast(Layer*)malloc(hiddenNumber * Layer.sizeof);
				for (uint j = 0; j < hiddenNumber; j++)
					hiddenLayers[j] = Layer(params.neurons, params.neurons, generator);
			}
	}
	
	unittest
	{
//		import std.stdio;
//		
//		writeln("Network.randomPopulation(ref Network* population, in NetworkParams params, in uint size)");
//		
//		NetworkParams params;
//		params.inputs  = 2;
//		params.layers  = 2;
//		params.neurons = 3;
//		params.outputs = 1;
//		
//		uint size = 1;
//		
//		Network* population;
//		scope(exit)
//		{
//			for(int i = 0; i < size; i++)
//				population[i].freeMem();
//			free(population);
//		}
//		
//		writeln(">>> Generating random population of ", size, " networks with parameters: ", params);
//		randomPopulation(population, params, size);
//		
//		float* input;  scope(exit) free(input);
//		float* hidden; scope(exit) free(hidden);
//		float* output; scope(exit) free(output);
//		
//		with(population[0])
//		{
//			input  = cast(float*)malloc(inputLayer.size);
//			hidden = cast(float*)malloc(hiddenLayer.size);
//			output = cast(float*)malloc(outputLayer.size);
//			
//			cudaMemcpy(input,  inputLayer,  inputLayer.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//			cudaMemcpy(hidden, hiddenLayer, hiddenLayer.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//			cudaMemcpy(output, outputLayer, outputLayer.size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//			
//			write(">>> Resulting input  layer = [");
//			for (uint i = 0; i < inputLayer.length; i++)
//				if (i != inputLayer.length - 1)
//					writef("% e, ", input[i]);
//				else
//					writefln("% e]", input[i]);
//			
//			write(">>> Resulting hidden layer = [");
//			for (uint i = 0; i < hiddenLayer.length; i++)
//				if (i != hiddenLayer.length - 1)
//					writef("% e, ", hidden[i]);
//				else
//					writefln("% e]", hidden[i]);
//			
//			write(">>> Resulting output layer = [");
//			for (uint i = 0; i < outputLayer.length; i++)
//				if (i != outputLayer.length - 1)
//					writef("% e, ", output[i]);
//				else
//					writefln("% e]", output[i]);
//		}
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

struct Data
{
	alias data this;
	
	float* data;         /// Pointer to the data array itself.
	uint   vectorLength; /// Size of an input vector.
	uint   dataLength;   /// Number of measurements.
}

struct Layer
{
	private struct WeightGroup
	{
		alias weights this;
		
		float* weights;
		
		uint weightsNumber;
		uint neuronsNumber;
		
		static immutable biasLength = 1; /// Number of bias weights per neuron.
		
		invariant
		{
			assert (weightsNumber >= 1);
			assert (neuronsNumber >= 1);
		}
		
		this(in uint weightsNumber, in uint neuronsNumber, ref curandGenerator_t generator) nothrow @nogc
		{
			this.weightsNumber = weightsNumber;
			this.neuronsNumber = neuronsNumber;
			
			cudaMalloc(weights, length);
			curandGenerate(generator, weights, length);
		}
		
		void freeMem() nothrow @nogc
		{
			if (weights !is null)
				cudaFree(weights);
		}
		
		@property ulong length() const pure nothrow @safe @nogc
		{
			return weightsNumber * neuronsNumber + biasLength;
		}
		
		/**
		 * Size of the weights array in bytes.
		 *
		 * Note:
		 *     Not to be confused with .sizeof property.
		 *     sizeof return size of the struct itself in the memory whether size() returns size of memory allocated for weights array.
		 */
		@property ulong size() const pure nothrow @safe @nogc
		{
			return length() * float.sizeof;
		}
		
		@property string toString() const
		{
			float* w; scope(exit) free(w);
			
			w = cast(float*)malloc(size);
			cudaMemcpy(w, weights, size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
			
			string result = "WeightGroup(weightsNumber = %d, neuronsNumber = %d, weights = [".format(weightsNumber, neuronsNumber);
			for (uint i = 0; i < length; i++)
				if (i == length - 1)
					result ~= "% e".format(w[i]);
				else
					result ~= "% e, ".format(w[i]);
			
			return result ~ "])";
		}
	}
	
	WeightGroup forward;   /// Array of connections weights for forwartd propagation.
	WeightGroup recurrent; /// Array of recurrent weights.
	WeightGroup backward;  /// Array of connections weights for backward propagation.
	
	uint neuronsNumber
	
	invariant
	{
		assert (&forward);
		assert (&recurrent);
		assert (&backward);
	}
	
	/**
	 * Default constructor.
	 *
	 * Params:
	 *     forwardNumber = Number of forward weights per neuron.
	 *     neuronsNumber = Number of neurons in the layer.
	 *     generator = Pseudorandom number generator.
	 */
	this(uint forwardNumber, uint neuronsNumber, curandGenerator_t generator) nothrow @nogc
	in
	{
		assert (forwardNumber >= 1);
		assert (neuronsNumber >= 1);
	}
	body
	{
		forward   = WeightGroup(forwardNumber, neuronsNumber, generator);
		recurrent = WeightGroup(neuronsNumber, neuronsNumber, generator);
		backward  = WeightGroup(neuronsNumber, neuronsNumber, generator);
	}
	
	/**
	 * Free memory.
	 */
	void freeMem() nothrow @nogc
	{
		forward.freeMem();
		recurrent.freeMem();
		backward.freeMem();
	}
	
	debug @property string toString() const
	{
		return "Layer(neuronsNumber = %d".format(neuro)
			~ ", forward = " ~ forward.toString()
			~ ", recurrent = " ~ recurrent.toString()
			~ ", backward = " ~ backward.toString()
			~ ")";
	}
}
	
unittest
{
	writeln("Layer.this(uint weightsPerNeuron, uint neuronsNum) nothrow @nogc");
	
	// Initialize cuRAND generator.
	curandGenerator_t generator;
	curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
	
	scope(exit) curandDestroyGenerator(generator);
	
	Layer l = Layer(3, 2, generator); scope(exit) l.freeMem();
	
	writeln(">>> Resulting layer = ", l.toString);
}

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

