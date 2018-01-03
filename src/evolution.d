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
import std.algorithm   : map;
import std.typecons    : Tuple;
import std.math        : cmp;
import std.array;

import statistics;

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
 * Genetare random layer chromosomes.
 *
 * Params:
 *     nNumber = Neurons number;
 *     wNumber = Number of connections.
 *     wBounds = Min and max possible connection weight.
 *     generator = (Pseudo)random number generator.
 */
double[][] randomGenes(T)(in ulong nNumber, in ulong wNumber, in WeightBounds wBounds, ref T generator)
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
		assert (n.length == wNumber + 1); // +1 goes for bias
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
		).take(wNumber + 1) // +1 goes for bias
		 .array;
	return result;
}

// Force contract call
unittest
{
	import std.random : Mt19937_64, unpredictableSeed;
	
	auto rng = Mt19937_64(unpredictableSeed());
	
	WeightBounds wb;
	wb.min = -10;
	wb.max =  10;
	
	double[][] g = randomGenes(5, 5, wb, rng);
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
	
	private
	{
		static immutable double crossoverRate = 0.5;  /// Determines probability of gene exchange.
		static immutable double alpha         = 0.9;  /// Determines weigth of gene exchange. x1 = (1 - alpha) * y1 | x2 = alpha * y2
		static immutable double mutationRate  = 0.05; /// Determines how often genes will mutate.
	}
	
	invariant
	{
		assert (crossoverRate >= 0.0 && crossoverRate <= 1.0);
		assert (alpha         >= 0.0 && alpha         <= 1.0);
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
		import std.random : Mt19937_64, unpredictableSeed;
		
		writeln("Genome.random(T)(in SpecimenParams params, ref T generator)");
		
		auto rng = Mt19937_64(unpredictableSeed());
		
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
	 *     generator = (Pseudo)random generator.
	 *                 Produces randomnes to chose how genes will be crossed over.
	 * Returns:
	 *     Array of exactly 2 genomes, which are results of crossing over 2 parant genomes.
	 */
	static Genome[2] crossover(T)(in Genome[2] parents, ref T generator)
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
		import std.random : Mt19937_64, unpredictableSeed;
		
		writeln("Genome.crossover(T)(in Genome[2] g, in ulong splitPoint, in double alpha, ref T generator)");
		
		auto rng = Mt19937_64(unpredictableSeed());
		
		SpecimenParams sp;
		sp.inputs  = 3;
		sp.outputs = 2;
		sp.layers  = 4;
		sp.neurons = 5;
		sp.weights.min = -10;
		sp.weights.max =  10;
		
		Genome[2] p = [random(sp, rng), random(sp, rng)];
		Genome[2] c = crossover(p, rng);
	}
	
	/**
	 * Mutates a genone.
	 *
	 * Randomly changes some genes, which are randomly selected too.
	 *
	 * Params:
	 *     sp = Specimen parameters. Applies the same restriction to mutations like
	 *          during genome generation.
	 *     generator = (Pseudo)random generator. Produces radnomnes to decide
	 *                 what genes and how they are going to mutate.
	 */
	void mutate(T)(in SpecimenParams sp, ref T generator)
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
		import std.random : Mt19937_64, unpredictableSeed;
		
		writeln("Genome.mutate(T)(in SpecimenParams sp, ref T generator)");
		
		auto rng = Mt19937_64(unpredictableSeed());
		
		SpecimenParams sp;
		sp.inputs  = 3;
		sp.outputs = 2;
		sp.layers  = 4;
		sp.neurons = 5;
		sp.weights.min = -10;
		sp.weights.max =  10;
		
		Genome g = random(sp, rng);
		g.mutate(sp, rng);
	}
}

struct Population
{
	alias Data    = Tuple!(double[], "input",  double[], "output");
//	alias Fitness = Tuple!(Genome,   "genome", double,   "fitness");
	
	SpecimenParams specimenParams;
	Genome[]       population;
	Data[]         trainingData;
	double[Genome] fitness;
	
	alias population this;
	
	/**
	 * Create initial population with random genomes.
	 *
	 * Params:
	 *     size = Numder of desired genomes.
	 *     generator = (Pseudo)random generator.
	 */
	void populate(U)(in ulong size, ref U generator)
	{
		population = generate(
			() => Genome.random(sp, generator)
		).take(size)
		 .array;
	}
	
	/**
	 * Default tournamed group size.
	 *
	 * Based on current population size.
	 */
	@property ulong tournamentSize()
	{
		return population.length / 10;
	}
	
	/**
	 * Measure and store fitnesses of all organisms.
	 */
	double[Genome] fitness(Genome[] sample)
	{
		double[Genome] result;
		
//		foreach (organism; sample)
//			result[organism] = 1 / MARE(
//				trainingData.map!(d => magnitude(         d.output)) .array,
//				trainingData.map!(d => magnitude(organism(d.input ))).array
//			);
		
		return result;
	}
	
	/**
	 * A tournament based selection.
	 *
	 * Params:
	 *     groupSize = Size of a random group to select.
	 *
	 * Returns:
	 *     The best genome from a random group.
	 */
	Genome tournament(string op)(ulong groupSize)
		if (op == "<" || op == ">")
	in
	{
		assert (groupSize >= 1 && groupSize <= population.length);
	}
	body
	{
		immutable string condition = "fitnessTable[organism]" ~ op ~ "fitnessTable[winner]";
		
		scope Genome[] group  = randomSample(population, groupSize).array;
		Genome winner = group[0];
		
		foreach (organism; group)
			if (mixin(condition))
			{
				winner = organism;
			}
			
		return winner;
	}
	
	/**
	 * Select two distinct parents.
	 *
	 * Params:
	 *     groupSize = Size of a random group to select.
	 *
	 * Returns:
	 *     Two parents' genomes based on their fitnesses.
	 */
	Genome[2] selectParents(ulong groupSize)
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
		
		result[0] = tournament!"<"(groupSize);
		do
		{
			result[1] = tournament!"<"(groupSize);
		}
		while (result[0] != result[1]);
		
		return result;
	}
	
	/**
	 * Create new subpopulation.
	 *
	 * Params:
	 *     amount = Size of a new population.
	 *     groupSize = Size of a tournament selection group. 
	 *     generator = (Pseudo)random generator.
	 *
	 * Returns:
	 *     Subpopulation of children of the most successeful parents.
	 */
	Genome[] breed(U)(ulong amount, ulong groupSize, ref U generator)
	{
		Genome[] result;
		for (ulong i = 0; i < size; i++)
			result ~= Genome.crossover(
				selectParents(groupSize),
				generator
			).each!mutate;
		
		return result;
	}
	
	/**
	 * Remove the least successeful orgamisms from popultion.
	 */
	void kill(U)(ulong amount, ref U generator)
	{
		for (ulong i = 0; i < size; i++)
		{
			T orgToKill = tournament!(Selection.worst)(tournamentSize);
			population.remove(orgToKill);
			fitnessTable.remove(orgToKill);
		}
	}
	
	void newGeneration(U)(ref U generator)
	{
		T[] newPop = breed(tournamentSize);
		kill(tournamentSize);
		
	}
}

