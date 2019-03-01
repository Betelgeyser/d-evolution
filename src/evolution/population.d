/**
 * Copyright © 2018 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
import std.algorithm : each, map, mean, sort, swap;
import std.math      : lround, isNaN;
import std.string : format;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common;
import math;
import neural.network;

public import neural.network : NetworkParams;

version (unittest)
{
	import std.algorithm : all, isSorted;
	import std.math      : isFinite;
	
	private RandomPool     randomPool;
	
	static this()
	{
		randomPool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, 0, 100_000);
	}
	
	static ~this()
	{
		randomPool.freeMem();
	}
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
		return
			this.fitness == i.fitness ||
			(isNaN(this.fitness) && isNaN(i.fitness));
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
	int opCmp()(auto ref in Individual i) const @safe
	{
		if (this.opEquals(i))
			return 0;
		else if (this.fitness > i.fitness || isNaN(i.fitness))
			return 1;
		else if (this.fitness < i.fitness || isNaN(this.fitness))
			return -1;
		else
			assert (0, "Float comparasion error. Could not compare %f and %f".format(this.fitness, i.fitness));
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
		Individual[] _individuals; /// All individuals of the population, including reserved;
		
		Individual[] _currentGeneration; /// Individuals of the current generation.
		Individual[] _newGeneration;     /// Memory reserved for offsprings which are the next generation.
		
		NetworkParams _networkParams; /// Stored network parameters.
		
		size_t _size;      /// Size of the population.
		ulong _generation; /// Current generation number.
		
		bool _isOrdered; /// Shows if individuals are already ordered.
		
		/**
		 * Size of the elite.
		 *
		 * Elite are individuals that are guaranteed to survive till the next generation. This is generally considered to improve
		 * GA convergence as saving the elites prevents good individuals from death.
		 */
		static immutable _elite = 10;
		
		float function(in Matrix, in Matrix, cublasHandle_t) _fitnessFunction;
	}
	
	invariant
	{
		if (_individuals.length)
		{
			assert (_currentGeneration.length == _newGeneration.length);
			assert (_individuals.length == _currentGeneration.length + _newGeneration.length);
		}
	}
	
	/**
	 * Returns: The number of the current generation, where 0 means initial generation.
	 */
	@property ulong generation() const @nogc nothrow pure @safe
	{
		return _generation;
	}
	
	/**
	 * Returns: Fitness of the best individual in the current generation.
	 */
	@property const(Individual) best() @safe
	{
		if (!_isOrdered)
			this._order();
		
		return _currentGeneration[$ - 1];
	}
	
	/**
	 * Returns: Fitness of the best individual in the current generation.
	 */
	@property float worst() @safe
	{
		if (!_isOrdered)
			this._order();
		
		return _currentGeneration[0].fitness;
	}
	
	/**
	 * Returns: Mean fitness of the current generation.
	 */
	@property float mean() const @nogc nothrow pure @safe
	{
		return _currentGeneration.map!"a.fitness".mean;
	}
	
	/**
	 * Random population constructor.
	 *
	 * Params:
	 *     params = Networks of the population are generated using this parameters.
	 *     size = Number of individuals in a single generation.
	 *     pool = Pool of random values.
	 *     fitnessFunction = Fitness tunction to use during the evolution process. MAE is default.
	 */
	this(in NetworkParams params, in ulong size, RandomPool pool, float function(in Matrix, in Matrix, cublasHandle_t) fitnessFunction = &math.MAE)
	in
	{
		assert (&params, "Incorrect network parameters");
		assert (size >= 100);
	}
	body
	{
		scope(failure) freeMem();
		
		_size          = size;
		_networkParams = params;
		_individuals   = nogcMalloc!Individual(size * 2);
		
		_currentGeneration = _individuals[0 .. size];
		_newGeneration     = _individuals[size .. $];
		
		_currentGeneration.each!((ref x) => x = Network(params, pool));
		_newGeneration.each!((ref x) => x = Network(params, pool)); // There is no need to initialize offsprings
		                                                            // in the first generation, but we need to allocate memory,
		                                                            // without that freeMem will fail.
		
		_fitnessFunction = fitnessFunction;
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		NetworkParams params = { inputs : 4, outputs : 2, neurons : 3, layers : 4 };
		immutable size = 100;
		
		auto population = Population(params, size, randomPool);
		scope(exit) population.freeMem();
		
		assert (population._currentGeneration.length == size);
		assert (population._newGeneration.length     == size);
		
		with (population)
		{
			// Not that population should test networks, but need to check whether population creates all networks or not
			assert (
				_currentGeneration.all!(
					i => i.individual.layers.all!(
						l => l.weights.all!(
							w => isFinite(w))))
			);
			assert (
				_currentGeneration.all!(
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
	void freeMem() nothrow
	{
		_individuals.each!(x => x.freeMem());
		
		if (_individuals.length)
			nogcFree(_individuals);
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
	void fitness(in Matrix inputs, in Matrix outputs, cublasHandle_t cublasHandle)
	{
		_isOrdered = false;
		
		auto approx = Matrix(outputs.rows, outputs.cols);
		scope(exit) approx.freeMem();
		
		foreach (ref individual; _currentGeneration)
		{
			individual(inputs, approx, cublasHandle);
			
			individual.fitness = _fitnessFunction(outputs, approx, cublasHandle);
		}
	}
	
	unittest
	{
		// Have no idea how to test this. Network activation and MASE themselves must be already tested at this point.
		mixin(notTested!fitness);
	}
	
	/**
	 * Order population by fitness values descending.
	 *
	 * As currently only MASE fitness function is supported and the lower it gets - the better it is, therefor the better
	 * an individual - the higher index it has. This desicion is made to ease implementation of the rank based selection.
	 */
	private void _order() @safe
	{
		if (!_isOrdered)
		{
			_currentGeneration.sort!"a > b"();
			_isOrdered = true;
		}
	}
	
	///
	unittest
	{
		mixin(writeTest!_order);
		
		NetworkParams params = { inputs : 5, outputs : 2, neurons : 4, layers : 5 };
		immutable size = 100;
		
		auto population = Population(params, size, randomPool);
		scope(exit) population.freeMem();
		cudaDeviceSynchronize();
		
		// Fill fitness values with random data
		foreach (ref i; population._currentGeneration)
			i.fitness = i.layers[0].weights[0];
		
		assert (!population._isOrdered);
		
		population._order();
		
		assert (population._currentGeneration.isSorted!"a.fitness > b.fitness");
		assert (population._isOrdered);
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
	void evolve(RandomPool pool)
	{
		this._order();
		
		immutable float ranksSum = AS(1, _size, _size);
		
		auto xParents = cudaMallocManaged!uint(_size - _elite);
		scope(exit) cudaFree(xParents);
		
		float[] randomScores = pool(_size - _elite).cudaScale(0, ranksSum);
		cudaRBS(xParents, randomScores);
		
		auto yParents = cudaMallocManaged!uint(_size - _elite);
		scope(exit) cudaFree(yParents);
		
		randomScores = pool(_size - _elite).cudaScale(0, ranksSum);
		cudaRBS(yParents, randomScores);
		
		cudaDeviceSynchronize();
		
		foreach (i; 0 .. _size - _elite)
			_newGeneration[i].crossover(
				_currentGeneration[xParents[i]],
				_currentGeneration[yParents[i]],
				_networkParams.min, _networkParams.max,
				0.5,
				pool
			);
		
		foreach (i; _size - _elite .. _size)
			Network.copy(_currentGeneration[i], _newGeneration[i]);
		
		swap(_currentGeneration, _newGeneration);
		
		_isOrdered = false;
		++_generation;
	}
	
	///
	unittest
	{
		mixin(notTested!evolve);
		
		auto pool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, 0, 10000);
		
		NetworkParams params = { inputs : 5, outputs : 1, neurons : 3, layers : 5, min : -1.0e3, max : 1.0e3 };
		immutable size = 100;
		
		auto population = Population(params, size, pool);
		scope(exit) population.freeMem();
		
		population.evolve(pool);
	}
}

