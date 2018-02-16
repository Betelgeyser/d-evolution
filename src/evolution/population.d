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
import cuda.curand;

// DNN modules
import common;
import neural.network;


struct Population
{
	ulong size;
	
	Network* population;
	
	this(in NetworkParams params, in ulong size, ref curandGenerator_t generator)// nothrow @nogc
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
		assert (p.population[0].depth == params.layers);
		assert (p.population[9].depth == params.layers);
		assert (p.population[9].hiddenLayers[params.layers - 1].length == (params.neurons + 1) * params.neurons);
	}
	
	void freeMem() nothrow @nogc
	{
		if (population !is null)
		{
			for (ulong i = 0; i < size; i++)
				population[i].freeMem();
			free(population);
		}
	}
}

