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
import std.algorithm : each, sort;
import std.math      : lround;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common;
import math;
import neural.network;


version (unittest)
{
	import std.algorithm : all, isSorted;
	import std.math      : isFinite;
}

/**
 * A single individual of a population paired with its fitness value.
 */
struct Individual
{
	alias individual this;
	
	Network individual; /// Individual.
	float   fitness;    /// Value of individual's fitness.
	
	/**
	 * Tests if two individuals have same fitness value.
	 */
	bool opEquals()(auto ref in Individual i) const @nogc nothrow pure @safe
	{
		return this.fitness == i.fitness;
	}
	
	unittest
	{
		mixin(writeTest!opEquals);
		
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
	int opCmp()(auto ref in Individual i) const @nogc nothrow pure @safe
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
		mixin(writeTest!opCmp);
		
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

/**
 * Rank selection only
 */
struct Population
{
	private
	{
		Individual[]  _individuals;
		Network[]     _offsprings;
		NetworkParams _networkParams;
	}
	
	float selectivePressure = 0.20; /// Determines what fraction of the population will be renewed every generation.
	
	invariant
	{
		assert (selectivePressure > 0 && selectivePressure <= 1); 
	}

	@property const(Individual[]) individuals() const @nogc nothrow pure @safe
	{
		return _individuals;
	}
	
	/**
	 * this
	 */
	this(in NetworkParams params, in ulong size, RandomPool pool) nothrow @nogc
	in
	{
		assert (&params, "Incorrect network parameters");
		assert (size >= 0);
	}
	out
	{
		assert (_individuals.length);
		assert (_offsprings.length, "Population must produce at least one offspring each generation.");
	}
	body
	{
		scope(failure) freeMem();
		
		_networkParams = params;
		
		_individuals = nogcMalloc!Individual(size);
		_offsprings  = nogcMalloc!Network(lround(size * selectivePressure));
		
		_individuals.each!((ref x) => x = Network(params, pool));
		_offsprings.each!((ref x) => x = Network(params, pool)); // There is no need to initialize offsprings
		                                                         // in the first generation, but we need to allocate memory,
		                                                         // without that freeMem will fail.
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		NetworkParams params = { inputs : 4, outputs : 2, neurons : 3, layers : 4 };
		immutable size = 10;
		
		auto population = Population(params, size, randomPool);
		scope(exit) population.freeMem();
		
		with (population)
		{
			assert (_individuals.length == size);
			assert (_offsprings.length  == lround(size * selectivePressure));
			
			// Not that population should test networks, but need to check whether population creates all networks or not
			assert (
				individuals.all!(
					i => i.individual.layers.all!(
						l => l.weights.all!(
							w => isFinite(w))))
			);
			assert (
				individuals.all!(
					i => i.individual.layers.all!(
						l => l.weights.all!(
							w => w.between(params.min, params.max))))
			);
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
		_individuals.each!(x => x.freeMem);
		_offsprings.each!(x => x.freeMem);
		
		if (_individuals.length)
			nogcFree(_individuals);
		if (_offsprings.length)
			nogcFree(_offsprings);
	}
	
	/**
	 * Calculates fitness of each individual in the population.
	 *
	 * Params:
	 *     inputs = Input matrix of a size m x n, where n is the number of input variables + 1 for bias and m is the number
	 *         of measurements.
	 *     outputs = Output matrix of a size m x k, where k is the number of output variables and m is the number of results.
	 *     cublasHandle = Cublas handle.
	 */
	void fitness(in Matrix inputs, in Matrix outputs, cublasHandle_t cublasHandle) nothrow @nogc
	{
		auto outputsT = Matrix(outputs.cols, outputs.rows); // MASE operates on transposed matrices
		auto approx   = Matrix(outputs.rows, outputs.cols); // Network's output is an approximated result
		auto approxT  = Matrix(outputs.cols, outputs.rows); // MASE operates on transposed matrices
		
		transpose(outputs, outputsT, cublasHandle);
		
		foreach (ref i; _individuals)
		{
			i(inputs, approx, cublasHandle);
			
			transpose(approx, approxT, cublasHandle);
			
			i.fitness = MASE(outputsT, approxT, cublasHandle);
		}
	}
	
	unittest
	{
		// Have no idea what to test here. Network activation and MASE themselves must be already tested at this point.
		mixin(notTested!fitness);
	}
	
	/**
	 * Order population by fitness values ascending.
	 *
	 * As currently only MASE fitness function is supported, the first individuals with lower fitness values are better ones.
	 */
	void order() nothrow @nogc
	{
		_individuals.sort!"a < b"();
	}
	
	///
	unittest
	{
		mixin(writeTest!order);
		
		NetworkParams params = { inputs : 5, outputs : 2, neurons : 4, layers : 5 };
		immutable size = 100;
		
		auto population = Population(params, size, randomPool);
		scope(exit) population.freeMem();
		cudaDeviceSynchronize();
		
		// Fill fitness values with random data
		foreach (ref i; population._individuals)
			i.fitness = i.layers[0].weights[0];
		
		population.order();
		
		assert (population._individuals.isSorted!"a.fitness < b.fitness");
	}
	
	/**
	 * Crossover population and create new generation of offsprings.
	 *
	 * Currently only the rank based selection is implemented.
	 *
	 * The rank based selection is similar to the roulette-wheel selection in a sense, but instead of chosing an individual
	 * proportionate to its fitness value in the RBS individuals are selected proportionate to theirs rank values.
	 *
	 * It is considered that RBS performes better global optimization while requires more time to converge.
	 *
	 * TODO: currently individual CAN breed with ITSELF! Despite chances are low, this is a bad practice,
	 * should be fixed later. That probably implies rewriting RandomPool fully on CUDA C++ 
	 */
	void breed(RandomPool pool)
	{
		this.order();
		
		immutable float ranksSum = AS(1, _individuals.length, _individuals.length);
		
		uint[] xParents;
		cudaMallocManaged(xParents, _offsprings.length);
		scope(exit) cudaFree(xParents);
		
		float[] randomScores = cudaScale(pool(_offsprings.length), 0, ranksSum);
		cudaRBS(xParents, randomScores);
		
		uint[] yParents;
		cudaMallocManaged(yParents, _offsprings.length);
		scope(exit) cudaFree(yParents);
		
		randomScores = cudaScale(pool(_offsprings.length), 0, ranksSum);
		cudaRBS(yParents, randomScores);
		
		foreach (i; 0 .. _offsprings.length)
			_offsprings[i].crossover(
				_individuals[xParents[i]],
				_individuals[yParents[i]],
				_networkParams.min, _networkParams.max,
				0.5,
				pool
			);
	}
	
	///
	unittest
	{
		mixin(notTested!breed);
		
		NetworkParams params = { inputs : 5, outputs : 1, neurons : 3, layers : 5, min : -1.0e3, max : 1.0e3 };
		immutable size = 10;
		
		auto population = Population(params, size, randomPool);
		scope(exit) population.freeMem();
		
		population.breed(randomPool);
	}
}

