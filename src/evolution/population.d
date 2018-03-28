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

// C modules
import core.stdc.stdlib;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
import cuda.cublas;

// DNN modules
import common;
import math;
import neural.network;

version (unittest)
{
	import std.random : unpredictableSeed;
	import std.math : approxEqual;
	
	private immutable accuracy = 0.000001;
	
	private curandGenerator_t generator;
	private cublasHandle_t handle;
	
	static this()
	{
		// Initialize cuRAND generator.
		curandCreateGenerator(generator, curandRngType_t.CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, 0);
		
		// Initialize cuBLAS
		cublasCreate(handle);
	}
	
	static ~this()
	{
		curandDestroyGenerator(generator);
		cublasDestroy(handle);
	}
}


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
	ulong size;
	
	Individual* individual;
	
	Matrix inputs;
	Matrix outputs;
	
	this(in NetworkParams params, in ulong size, curandGenerator_t generator) nothrow @nogc
	{
		scope(failure) freeMem();
		
		this.size = size;
		
		individual = cast(Individual*)malloc(size * Individual.sizeof);
		for (ulong i = 0; i < size; i++)
			individual[i] = Network(params, generator);
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
		curandSetPseudoRandomGeneratorSeed(generator, unpredictableSeed());
		
		scope(exit) curandDestroyGenerator(generator);
		
		Population p = Population(params, 10, generator); scope(exit) p.freeMem();
		
		assert (p.size == 10);
		
		// Check memory
		assert (p.individual[0].depth          == params.layers);
		assert (p.individual[p.size - 1].depth == params.layers);
		assert (
			p.individual[p.size - 1].hiddenLayers[params.layers - 1].weights.length
			== (params.neurons + 1) * params.neurons
		);
	}
	
	void freeMem() nothrow @nogc
	{
		if (size > 0)
		{
			for (ulong i = 0; i < size; i++)
				individual[i].freeMem();
			free(individual);
		}
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
		
		for (uint i = 0; i < size; i++)
		{
			individual[i](inputs, approx, cublasHandle);
			
			transpose(approx, approx_t, cublasHandle);
			
			individual[i].fitness = MASE(outputs_t, approx_t, cublasHandle);
		}
	}
	
	unittest
	{
		mixin(notTested!evaluate);
	}
}

