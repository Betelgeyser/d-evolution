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

import std.random      : uniform, randomSample;
import std.range       : generate, take;
import std.conv        : to;
import std.parallelism : parallel;
//import std.algorithm   : map, sum, minElement, maxElement;
import std.typecons    : Tuple;
import std.math        : lrint;
import std.container;
import std.array;

import evolution.genome;
//import dnn.math;

struct Population(T)
{
	alias Data = Tuple!(double[], "input",  double[], "output");
	
	/**
	 * Parameters to generate new genomes.
	 */
	SpecimenParams specimenParams;
	
	double crossoverRate = 0.50; /// Determines probability of gene exchange.
	double alpha         = 0.90; /// Determines weigth of gene exchange. x1 = (1 - alpha) * y1 | x2 = alpha * y2
	double mutationRate  = 0.05; /// Determines how often genes will mutate.
	
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
		Array!Data trainingData;
		
		/**
		 * Genomes.
		 *
		 * Cunny hack.
		 */
		Array!Genome population;

		/**
		 * Default tournament group size.
		 *
		 * Based on current population size.
		 */
		@property ulong tournamentSize() const pure nothrow @safe @nogc
		out (result)
		{
			assert (result >= 2);
		}
		body
		{
			return lrint(population.length * 0.5);
		}
		
		/**
		 * Size of new generations.
		 *
		 * Based on current population size.
		 */
		@property ulong breedSize() const pure nothrow @safe @nogc
		out (result)
		{
			assert (result >= 2);
			assert (result % 2 == 0);
		}
		body
		{
			return lrint(population.length * 0.2);
		}
		
		/**
		 * Evaluate fitness of genome.
		 *
		 * Params:
		 *     genome = Genome to measure fitness on.
		 */
//		double evaluate(Genome genome) const pure nothrow @safe
//		{
//			return MARE(
//				trainingData.map!(d =>           d.output).array,
//				trainingData.map!(d => T(genome)(d.input)).array
//			);
//		}
		
		/**
		 * A tournament based selection.
		 *
		 * Params:
		 *     groupSize = Size of a random group to select.
		 *
		 * Returns:
		 *     The best genome from a random group.
		 */
//		Genome tournament(string op)(ulong groupSize)
//			if (op == "<" || op == ">")
//		in
//		{
//			assert (groupSize >= 1 && groupSize <= population.length);
//		}
//		body
//		{
//			immutable string condition = "population[individual].fitness" ~ op ~ "population[winner].fitness";
//			
//			scope Genome[] group = randomSample(population.values.map!"a.genome", groupSize).array;
//			Genome winner = group[0];
//			
//			foreach (individual; group)
//				if (mixin(condition))
//					winner = individual;
//			
//			return winner;
//		}
		
		/**
		 * Select two distinct parents.
		 *
		 * Params:
		 *     groupSize = Size of a random group to select.
		 *
		 * Returns:
		 *     Two parent genomes based on their fitnesses.
		 */
//		Genome[2] selectParents(ulong groupSize)
//		in
//		{
//			assert (groupSize >= 1 && groupSize <= population.length);
//		}
//		out (result)
//		{
//			assert (result[0] != result[1]);
//		}
//		body
//		{
//			Genome[2] result;
//			
//			result[0] = tournament!"<"(groupSize);
//			do
//			{
//				result[1] = tournament!"<"(groupSize);
//			}
//			while (result[0] == result[1]);
//			
//			return result;
//		}
		
		/**
		 * Create new subpopulation.
		 *
		 * Params:
		 *     amount = Size of a new population.
		 *     groupSize = Size of a tournament selection group.
		 *
		 * Returns:
		 *     Children of the most successeful parents.
		 */
//		Genome[] breed(ulong amount, ulong groupSize)
//		in
//		{
//			assert (amount % 2 == 0);
//		}
//		body
//		{
//			Genome[] result;
//			for (ulong i = 0; i < amount / 2; i++)
//			{
//				result ~= Genome.crossover(
//					selectParents(groupSize),
//					crossoverRate,
//					alpha
//				);
//				
//				result[$ - 1].mutate(specimenParams, mutationRate);
//				result[$ - 2].mutate(specimenParams, mutationRate);
//			}
//			
//			return result;
//		}
		
		/**
		 * Remove the least successeful orgamisms from popultion.
		 *
		 * Params:
		 *     amount = How many individuals to kill.
		 *     groupSize = Size of a tournament selection group. 
		 *     generator = (Pseudo)random generator.
		 *                 Is required to select random tournament group.
		 */
//		void kill(in ulong amount, in ulong groupSize)
//		{
//			for (ulong i = 0; i < amount; i++)
//			{
//				Genome genomeToDie = tournament!">"(groupSize);
//				population.remove(genomeToDie);
//			}
//		}
	}
	
//	/**
//	 * Best fitness of the population.
//	 */
//	@property double bestFitness() const pure nothrow
//	{
//		return population.values.map!"a.fitness".minElement;
//	}
//	
//	/**
//	 * Best fitness of the population.
//	 */
//	@property double worstFitness() const pure nothrow
//	{
//		return population.values.map!"a.fitness".maxElement;
//	}
//	
//	/**
//	 * Average fitness of the population.
//	 */
//	@property double avgFitness() const pure nothrow
//	{
//		return population.values.map!"a.fitness".sum / population.length;
//	}
	
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
		trainingData.reserve(inputs.length);
//		foreach (ref data; trainingData)
//		{
//			data.input  = inputs [i].dup;
//			data.output = outputs[i].dup;
//		}
	}
	
	/**
	 * Create initial population with random genomes.
	 *
	 * Params:
	 *     size = Desired genomes number.
	 */
	void populate(in ulong size)
	{
		for (ulong i = 0; i < size; i++)
		{
			Genome individual = Genome.random(specimenParams);
			population[individual] = Individual(individual, evaluate(individual));
		}
	}
	
	void selection()
	{
		Genome[] newGeneration = breed(breedSize, tournamentSize);
		kill(breedSize, tournamentSize);
		
		foreach (individual; newGeneration)
			population[individual] = Individual(individual, evaluate(individual));
	}
}

