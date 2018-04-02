/**
 * Copyright © 2018 Sergei Iurevich Filippov, All Rights Reserved.
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
module evolution.population;

// D modules
import std.algorithm : sort;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import common;
import math;
import neural.network;

/**
 * A single individual ot a population stored with its fitness value.
 */
struct Individual
{
	alias individual this;
	
	Network individual; /// Individual.
	float fitness;      /// Value of individual's fitness. 
	
	/**
	 * Tests if two individuals have same fitness value. 
	 */
	bool opEquals()(auto ref scope const Individual i) const pure nothrow @safe @nogc
	{
		return this.fitness == i.fitness;
	}
	
	unittest
	{
		mixin(writetest!opEquals);
		
		Individual i1;
		Individual i2;
		Individual i3;
		
		i1.fitness =  0;
		i2.fitness =  0;
		i3.fitness = -0.000_000_1;
		
		assert (i1 == i2);
		assert (i1 != i3);
	}
	
	/**
	 * Compares fitness values of two individuals.
	 */
	int opCmp(in ref Individual i) const pure nothrow @safe @nogc
	{
		if (this.opEquals(i))
			return 0;
		else if (this.fitness > i.fitness)
			return 1;
		else if (this.fitness < i.fitness)
			return -1;
		else
			assert (0, "float comparasion error.");
	}
	
	unittest
	{
		mixin(writetest!opCmp);
		
		Individual i1;
		Individual i2;
		Individual i3;
		
		i1.fitness =  0;
		i2.fitness =  0;
		i3.fitness = -0.000_000_1;
		
		assert (i1 > i3);
		assert (i3 < i1);
		assert (i1 >= i2);
		assert (i2 >= i3);
	}
}

struct Population
{
	Individual[] individual;
	
	Matrix inputs;
	Matrix outputs;
	
	@property size() const pure nothrow @safe @nogc
	{
		return individual.length;
	}
	
	this(in NetworkParams params, in ulong size, curandGenerator_t generator) nothrow @nogc
	{
		scope(failure) freeMem();
		
		individual = nogcMalloc!Individual(size);
		foreach (ref i; individual)
			i = Network(params, generator);
	}
	
	///
	unittest
	{
		mixin(writetest!__ctor);
		
		// Initialize cuRAND generator
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		scope(exit) curandDestroyGenerator(generator);
		
		// Initialize network params
		NetworkParams params;
		params.inputs  = 2;
		params.outputs = 1;
		params.neurons = 3;
		params.layers  = 2;
		
		immutable size = 10;
		
		auto p = Population(params, size, generator);
		scope(exit) p.freeMem();
		
		assert (p.size == size);
		
		// Check memory
		assert (p.individual[0].depth          == params.layers);
		assert (p.individual[p.size - 1].depth == params.layers);
		assert (
			p.individual[p.size - 1].hiddenLayers[params.layers - 1].weights.length
			== (params.neurons + 1) * params.neurons
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
	void freeMem() nothrow @nogc
	{
		foreach (ref i; individual)
			i.freeMem();
		
		if (size > 0)
			free(individual);
	}
	
	/**
	 * Evaluate the population.
	 *
	 * Evaluates a result of feeding inpit matrix to the network.
	 *
	 * Params:
	 *     inputs = Input matrix of a size m x n, where n is the number of input variables + 1 for bias and m is the number
	 *         of measurements.
	 *     outputs = Output matrix of a size m x k, where k is the number of output variables and m is the number of results.
	 *     cublasHandle = Cublas handle.
	 */
	void evaluate(in Matrix inputs, in Matrix outputs, cublasHandle_t cublasHandle) nothrow @nogc
	{
		auto outputs_t = Matrix(outputs.cols, outputs.rows);
		auto approx    = Matrix(outputs.rows, outputs.cols);
		auto approx_t  = Matrix(outputs.cols, outputs.rows);
		
		transpose(outputs, outputs_t, cublasHandle);
		
		foreach (i; individual)
		{
			i(inputs, approx, cublasHandle);
			
			transpose(approx, approx_t, cublasHandle);
			
			i.fitness = MASE(outputs_t, approx_t, cublasHandle);
		}
	}
	
	unittest
	{
		mixin(notTested!evaluate);
	}
	
	void sort() nothrow @nogc
	{
		individual.sort!"a < b"();
	}
	
	///
	unittest
	{
		mixin(writetest!sort);
		
		// Initialize cuRAND generator
		curandGenerator_t generator;
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		scope(exit) curandDestroyGenerator(generator);
		
		// Initialize network params
		NetworkParams params;
		params.inputs  = 2;
		params.outputs = 1;
		params.neurons = 3;
		params.layers  = 2;
		
		immutable size = 10;
		
		auto p = Population(params, size, generator);
		cudaDeviceSynchronize();
		
		// Fill fitness values with random data
		for (ulong i = 0; i < size; ++i)
			p.individual[i].fitness = p.individual[i].inputLayer.weights[0];	
		
		p.sort();
		for (ulong i = 0; i < size - 1; ++i)
			assert (p.individual[i].fitness <= p.individual[i + 1].fitness);
	}
}

