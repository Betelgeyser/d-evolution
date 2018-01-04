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
module evolution;

import std.random      : uniform, randomSample;
import std.range       : generate, take;
import std.conv        : to;
import std.parallelism : parallel;
import std.algorithm   : map, sum, minElement, maxElement;
import std.typecons    : Tuple;
import std.math        : lrint;
import std.array;

import statistics;

immutable biasLength = 1;

/** Min and max weight value of a neuron connection. */
struct WeightBounds
{
	double min;
	double max;
	
	invariant
	{
		assert (min <= max);
	}
}

/**
 * Random genome generation parameters.
 */
struct SpecimenParams
{
	ulong inputs;  /// Number of network's inputs.
	ulong outputs; /// Number of network's outputs.
	ulong layers;  /// Number of hidden layers.
	ulong neurons; /// Number of neurons in every hidden layer.
	
	WeightBounds weights; /// Min and max weight value of a neuron connection.
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (layers  >= 1);
		assert (neurons >= 1);
		assert (&weights);
	}
}

/**
 * Struct that represents network genome.
 */
struct Genome
{
	/**
	 * Input layer chomosome.
	 *
	 * Represents number of input neurons.
	 */
	ulong input;
	
	/**
	 * Hidden and output layers chromosomes.
	 *
	 * The first index is a layer.
	 * The second one is a neuron.
	 * And the third one is a connection weight and bias.
	 *
	 * The last layer is the output layer.
	 */
	double[][][] hidden;
	
	/**
	 * Genetare random layer chromosomes.
	 *
	 * Params:
	 *     nNumber = Neurons number;
	 *     wNumber = Number of connections.
	 *     wBounds = Min and max possible connection weight.
	 *     generator = (Pseudo)random number generator.
	 */
	private static double[][] randomGenes(T)(in ulong nNumber, in ulong wNumber, in WeightBounds wBounds, ref T generator)
	in
	{
		assert (wNumber >= 1);
		assert (&wBounds);
	}
	out (result)
	{
		assert (result.length == nNumber);
		
		foreach (n; result)
		{
			assert (n.length == wNumber + biasLength);
			foreach (w; n)
				assert (w >= wBounds.min && w <= wBounds.max);
		}
	}
	body
	{
		double[][] result;
		for (long i = 0; i < nNumber; i++)
			result ~= generate(
				() => uniform!"[]"(wBounds.min, wBounds.max, generator)
			).take(wNumber + biasLength)
			 .array;
		return result;
	}
	
	// Force contract call
	unittest
	{
		import std.random : Mt19937_64;
		
		auto rng = Mt19937_64(0);
		
		WeightBounds wb;
		wb.min = -10;
		wb.max =  10;
		
		double[][] g = Genome.randomGenes(5, 5, wb, rng);
	}
	
	/**
	 * Generate random genome.
	 *
	 * Params:
	 *     params = Parameters of generated network specimen.
	 *     generator = (Pseudo)random number generator.
	 */
	static Genome random(T)(in SpecimenParams params, ref T generator)
	in
	{
		assert (&params);
	}
	out (result)
	{
		assert (result.hidden.length == params.layers + 1);
		foreach (i, l; result.hidden)
		{
			if (i == result.hidden.length - 1)
				assert (l.length == params.outputs);
			else
				assert (l.length == params.neurons);
			
			foreach (n; l)
			{
				if (i == 0)
					assert (n.length == params.inputs + 1);
				else
					assert (n.length == params.neurons + 1);
				
				foreach (w; n)
					assert (w >= params.weights.min && w <= params.weights.max);
			}
		}
	}
	body
	{
		Genome result;
		
		result.input = params.inputs;
		
		// Generate the first hidden layer
		result.hidden ~= randomGenes(params.neurons, params.inputs, params.weights, generator);
		
		// Generating remaining hidden layers
		for (ulong i = 0; i < params.layers - 1; i++)
			result.hidden ~= randomGenes(params.neurons, params.neurons, params.weights, generator);
		
		// Output layer
		result.hidden ~= randomGenes(params.outputs, params.neurons, params.weights, generator);
		
		return result;
	}
	
	// Force contract ckeck
	unittest
	{
		import std.stdio;
		import std.random : Mt19937_64;
		
		writeln("Genome.random(T)(in SpecimenParams params, ref T generator)");
		
		auto rng = Mt19937_64(0);
		
		SpecimenParams sp;
		sp.inputs  = 3;
		sp.outputs = 2;
		sp.layers  = 4;
		sp.neurons = 5;
		sp.weights.min = -10;
		sp.weights.max =  10;
		
		Genome g = random(sp, rng);
	}
	
	/**
	 * Crossover 2 genomes.
	 *
	 * Params:
	 *     parents = A pair of parents' genomes to crossover.
	 *     crossoverRate = Determines probability of gene exchange.
	 *     alpha = Determines weigth of gene exchange. x1 = (1 - alpha) * y1 | x2 = alpha * y2
	 *     generator = (Pseudo)random generator.
	 *                 Produces randomnes to chose how genes will be crossed over.
	 *
	 * Returns:
	 *     Array of exactly 2 genomes, which are results of crossing over 2 parant genomes.
	 */
	static Genome[2] crossover(T)(in Genome[2] parents, in double crossoverRate, in double alpha, ref T generator)
	in
	{
		assert (&parents[0]);
		assert (&parents[1]);
		
		assert (parents[0].input         == parents[1].input        );
		assert (parents[0].hidden.length == parents[1].hidden.length);
		
		foreach (li, layer; parents[0].hidden)
		{
			assert (layer.length == parents[1].hidden[li].length);
			foreach (ni, neuron; layer)
				assert (neuron.length == parents[1].hidden[li][ni].length);
		}
	}
	out (result)
	{
		assert (&result[0]);
		assert (&result[1]);
	}
	body
	{
		Genome[2] children;
		
		children[0].input = parents[0].input;
		children[1].input = parents[1].input;
		
		children[0].hidden.length = parents[0].hidden.length;
		children[1].hidden.length = parents[1].hidden.length;
		
		for (ulong li = 0; li < parents[0].hidden.length; li++)
		{
			children[0].hidden[li].length = parents[0].hidden[li].length;
			children[1].hidden[li].length = parents[1].hidden[li].length;
			
			for (ulong ni = 0; ni < parents[0].hidden[li].length; ni++)
			{
				children[0].hidden[li][ni].length = parents[0].hidden[li][ni].length;
				children[1].hidden[li][ni].length = parents[1].hidden[li][ni].length;
				
				for (ulong wi = 0; wi < parents[0].hidden[li][ni].length; wi++)
				{
					double roll   = uniform!"[)"(0.0, 1.0, generator);
					ulong  first  = (roll <  crossoverRate).to!ulong;
					ulong  second = (roll >= crossoverRate).to!ulong;
					
					children[first ].hidden[li][ni][wi] = parents[0].hidden[li][ni][wi] *      alpha;
					children[second].hidden[li][ni][wi] = parents[0].hidden[li][ni][wi] * (1 - alpha);
				}
			}
		}
		
		return children;
	}
	
	// Don't know how to properly test this...
	unittest
	{
		import std.stdio : writeln;
		import std.random : Mt19937_64;
		
		writeln("Genome.crossover(T)(in Genome[2] parents, ref T generator)");
		
		auto rng = Mt19937_64(0);
		
		SpecimenParams sp;
		sp.inputs  = 3;
		sp.outputs = 2;
		sp.layers  = 4;
		sp.neurons = 5;
		sp.weights.min = -10;
		sp.weights.max =  10;
		
		Genome[2] p = [random(sp, rng), random(sp, rng)];
		Genome[2] c = crossover(p, 0.5, 0.9, rng);
		
		writeln(">>> Parent[0] = ", p[0]);
		writeln(">>> Parent[1] = ", p[1]);
		writeln(">>> Child [0] = ", c[0]);
		writeln(">>> Child [1] = ", c[1]);
	}
	
	/**
	 * Mutates a genone.
	 *
	 * Randomly changes some genes, which are randomly selected too.
	 *
	 * Params:
	 *     sp = Specimen parameters. Applies the same restriction to mutations like
	 *          during genome generation.
	 *     mutationRate = Determines how often genes will mutate.
	 *     generator = (Pseudo)random generator. Produces radnomnes to decide
	 *                 what genes and how they are going to mutate.
	 */
	void mutate(T)(in SpecimenParams sp, in double mutationRate, ref T generator)
	in
	{
		assert (&sp);
		assert (&this);
	}
	body
	{
		foreach (layer; this.hidden)
			foreach (neuron; layer)
				foreach (ref weight; neuron)
					if (uniform!"[)"(0.0, 1.0, generator) < mutationRate)
						weight = uniform!"[]"(sp.weights.min, sp.weights.max, generator);
	}
	
	// Don't know how to properly test this...
	unittest
	{
		import std.stdio : writeln;
		import std.random : Mt19937_64;
		
		writeln("Genome.mutate(T)(in SpecimenParams sp, ref T generator)");
		
		auto rng = Mt19937_64(0);
		
		SpecimenParams sp;
		sp.inputs  = 3;
		sp.outputs = 2;
		sp.layers  = 4;
		sp.neurons = 5;
		sp.weights.min = -10;
		sp.weights.max =  10;
		
		Genome g = random(sp, rng);
		writeln("Before mutation = ", g);
		g.mutate(sp, 0.05, rng);
		writeln("After mutation  = ", g);
	}
}

struct Population(T)
{
	alias Data = Tuple!(double[], "input",  double[], "output");
	
	/**
	 * Parameters to generate new genomes.
	 */
	SpecimenParams specimenParams;
	
	double crossoverRate = 0.90; /// Determines probability of gene exchange.
	double alpha         = 0.90; /// Determines weigth of gene exchange. x1 = (1 - alpha) * y1 | x2 = alpha * y2
	double mutationRate  = 0.30; /// Determines how often genes will mutate.
	
	invariant
	{
		assert (crossoverRate >= 0.0 && crossoverRate <= 1.0);
		assert (alpha         >= 0.0 && alpha         <= 1.0);
		assert (mutationRate  >= 0.0 && mutationRate  <= 1.0);
	}
	
	private
	{
		/**
		 * Data to train on.
		 */
		Data[] trainingData;
		
		/**
		 * Genomes.
		 */
		Genome[] population;
		
		/**
		 * Fitnesses of genomes.
		 */
		double[Genome] fitness;

		/**
		 * Default tournament group size.
		 *
		 * Based on current population size.
		 */
		@property ulong tournamentSize() const pure nothrow @safe @nogc
		{
			return lrint(population.length * 0.5);
		}
		
		/**
		 * Size of new generations.
		 *
		 * Based on current population size.
		 */
		@property ulong breedSize() const pure nothrow @safe @nogc
		{
			return lrint(population.length * 0.2);
		}
		
		/**
		 * Evaluate fitness of genome.
		 *
		 * Params:
		 *     genome = Genome to measure fitness on.
		 */
		double evaluate(Genome genome) const pure nothrow @safe
		{
			return MARE(
				trainingData.map!(d =>           d.output).array,
				trainingData.map!(d => T(genome)(d.input)).array
			);
		}
		
		/**
		 * A tournament based selection.
		 *
		 * Params:
		 *     groupSize = Size of a random group to select.
		 *     generator = (Pseudo)random generator.
		 *                 Is required to select a random tournament group.
		 *
		 * Returns:
		 *     The best genome from a random group.
		 */
		Genome tournament(string op, U)(ulong groupSize, ref U generator)
			if (op == "<" || op == ">")
		in
		{
			assert (groupSize >= 1 && groupSize <= population.length);
		}
		body
		{
			immutable string condition = "fitness[individual]" ~ op ~ "fitness[winner]";
			
			scope Genome[] group = randomSample(population, groupSize, &generator).array;
			Genome winner = group[0];
			
			foreach (individual; group)
				if (mixin(condition))
					winner = individual;
			
			return winner;
		}
		
		/**
		 * Select two distinct parents.
		 *
		 * Params:
		 *     groupSize = Size of a random group to select.
		 *     generator = (Pseudo)random generator.
		 *                 Is required to select a random tournament group.
		 *
		 * Returns:
		 *     Two parent genomes based on their fitnesses.
		 */
		Genome[2] selectParents(U)(ulong groupSize, ref U generator)
		in
		{
			assert (groupSize >= 1 && groupSize <= population.length);
		}
		out (result)
		{
			assert (result[0] != result[1]);
		}
		body
		{
			Genome[2] result;
			
			result[0] = tournament!"<"(groupSize, generator);
			do
			{
				result[1] = tournament!"<"(groupSize, generator);
			}
			while (result[0] == result[1]);
			
			return result;
		}
		
		/**
		 * Create new subpopulation.
		 *
		 * Params:
		 *     amount = Size of a new population.
		 *     groupSize = Size of a tournament selection group. 
		 *     generator = (Pseudo)random generator.
		 *                 Is required to select random tournament group
		 *                 and to provide random mutations.
		 *
		 * Returns:
		 *     Children of the most successeful parents.
		 */
		Genome[] breed(U)(ulong amount, ulong groupSize, ref U generator)
		{
			Genome[] result;
			for (ulong i = 0; i < amount; i++)
			{
				result ~= Genome.crossover(
					selectParents(groupSize, generator),
					crossoverRate,
					alpha,
					generator
				);
				
				result[$ - 1].mutate(specimenParams, mutationRate, generator);
				result[$ - 2].mutate(specimenParams, mutationRate, generator);
			}
			
			return result;
		}
		
		/**
		 * Remove the least successeful orgamisms from popultion.
		 *
		 * Params:
		 *     amount = How many individuals to kill.
		 *     groupSize = Size of a tournament selection group. 
		 *     generator = (Pseudo)random generator.
		 *                 Is required to select random tournament group.
		 */
		void replace(U)(Genome[] newPopulation, ulong groupSize, ref U generator)
		{
			foreach (newIndiv; newPopulation)
			{
				Genome genomeToDie = tournament!">"(groupSize, generator);
				foreach (i, ref individual; population)
					if (individual == genomeToDie)
						individual = newIndiv;
				fitness.remove(genomeToDie);
				fitness[newIndiv] = evaluate(newIndiv);
			}
		}
	}
	
	/**
	 * Best fitness of the population.
	 */
	@property double bestFitness() const pure nothrow
	{
		return fitness.values.minElement;
	}
	
	/**
	 * Best fitness of the population.
	 */
	@property double worstFitness() const pure nothrow
	{
		return fitness.values.maxElement;
	}
	
	/**
	 * Average fitness of the population.
	 */
	@property double avgFitness() const pure nothrow
	{
		return fitness.values.sum / fitness.length;
	}
	
	/**
	 * Load data.
	 */
	void loadData(in double[][] inputs, in double[][] outputs) pure nothrow @safe
	in
	{
		assert (inputs.length == outputs.length);
	}
	body
	{
		for (ulong i = 0; i < inputs.length; i++)
		{
			trainingData ~= Data.init;
			trainingData[$ - 1].input  = inputs [i].dup;
			trainingData[$ - 1].output = outputs[i].dup;
		}
	}
	
	/**
	 * Create initial population with random genomes.
	 *
	 * Params:
	 *     size = Numder of desired genomes.
	 *     generator = (Pseudo)random generator.
	 *                 Is required for generating random genomes.
	 */
	void populate(U)(in ulong size, ref U generator)
	{
		population.length = size;
		
		foreach (ref individual; population)
		{
			individual = Genome.random(specimenParams, generator);
			fitness[individual] = evaluate(individual);
		}
	}
	
	void selection(U)(ref U generator)
	{
		Genome[] newGeneration = breed(breedSize, tournamentSize, generator);
		replace(newGeneration, tournamentSize, generator);
	}
}

