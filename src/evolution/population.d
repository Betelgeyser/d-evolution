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

// D modules
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
import neural.network;


struct Population
{
	ulong size;
	
	Network* population;
	
	Matrix inputs;
	Matrix outputs;
	
	this(in NetworkParams params, in ulong size, curandGenerator_t generator) nothrow @nogc
	{
		scope(failure) freeMem();
		
		this.size = size;
		
		population = cast(Network*)malloc(size * Network.sizeof);
		for (ulong i = 0; i < size; i++)
			population[i] = Network(params, generator);
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
		
		Population p = Population(params, 10, generator); scope(exit) p.freeMem();
		
		assert (p.size == 10);
		
		// Check memory
		assert (p.population[0].depth          == params.layers);
		assert (p.population[p.size - 1].depth == params.layers);
		assert (
			p.population[p.size - 1].hiddenLayers[params.layers - 1].weights.length
			== (params.neurons + 1) * params.neurons
		);
	}
	
	void freeMem() nothrow @nogc
	{
		if (size > 0)
		{
			for (ulong i = 0; i < size; i++)
				population[i].freeMem();
			free(population);
		}
	}
	
	/**
	 * Evaluate the population.
	 *
	 * Evaluates a result of feeding inpit matrix to the network.
	 *
	 * Params:
	 *     inputs = Input matrix of size m x k, where k is the number of neuron connections (incl. bias).
	 *     outputs = Output matrix of size m x n, where n is the number of output neurons.
	 *     cublasHandle = Cublas handle.
	 */
	void evaluate(in Matrix inputs, in Matrix outputs) nothrow @nogc
	{
		cublasHandle_t handle;
		cublasCreate(handle);
		scope(exit) cublasDestroy(handle);
		
		evaluate(inputs, outputs, handle);
	}
	
	/// ditto
	void evaluate(in Matrix inputs, in Matrix outputs, cublasHandle_t cublasHandle) nothrow @nogc
	{
		auto approximation = Matrix(outputs.rows, outputs.cols);
		
		for (uint i = 0; i < size; i++)
		{
			population[i](inputs, approximation, cublasHandle);
			population[i].fitness = MASE(
				outputs,
				approximation,
				cublasHandle
			);
		}
	}
	
	///
	unittest
	{
		mixin(writetest!evaluate);
		
		/* 1  2 *
		 * 3 -1 */
	}
}

