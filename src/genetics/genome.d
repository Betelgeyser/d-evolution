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
module genetic.genome;

import core.stdc.stdlib;

import cuda.cudaruntimeapi;

immutable ubyte biasLength = 1;


//
///**
// * Struct that represents network genome.
// */

/// Layer
struct Chromosome
{
	private
	{
		float* genes;
		uint   weights;
		uint   neurons;
	} 
	
}

struct Genome
{
	/**
	 * Input layer chomosome.
	 *
	 * Represents number of input neurons.
	 */
//	ulong input;
//	
//	/**
//	 * Fitness of this genome.
//	 */
//	double fitness;
//	
//	/**
//	 * Hidden and output layers chromosomes.
//	 *
//	 * The first index is a layer.
//	 * The second one is a neuron.
//	 * And the third one is a connection weight and bias.
//	 *
//	 * The last layer is the output layer.
//	 */
	Chromosome input;
	Chromosome hidden;
	Chromosome output;
	
	@disable this();
	

	
//	/**
//	 * Genetare random layer chromosomes.
//	 *
//	 * Params:
//	 *     nNumber = Neurons number;
//	 *     wNumber = Number of connections.
//	 *     wBounds = Min and max possible connection weight.
//	 *     generator = (Pseudo)random number generator.
//	 */
//	private static double[][] randomGenes(in ulong nNumber, in ulong wNumber, in WeightBounds wBounds)
//	in
//	{
//		assert (wNumber >= 1);
//		assert (&wBounds);
//	}
//	out (result)
//	{
//		assert (result.length == nNumber);
//		
//		foreach (n; result)
//		{
//			assert (n.length == wNumber + biasLength);
//			foreach (w; n)
//				assert (w >= wBounds.min && w <= wBounds.max);
//		}
//	}
//	body
//	{
//		double[][] result;
//		for (long i = 0; i < nNumber; i++)
//			result ~= generate(
//				() => uniform!"[]"(wBounds.min, wBounds.max)
//			).take(wNumber + biasLength)
//			 .array;
//		return result;
//	}
//	
//	// Force contract call
//	unittest
//	{
//		WeightBounds wb;
//		wb.min = -10;
//		wb.max =  10;
//		
//		double[][] g = Genome.randomGenes(5, 5, wb);
//	}
//	
//	/**
//	 * Generate random genome.
//	 *
//	 * Params:
//	 *     params = Parameters of generated network specimen.
//	 *     generator = (Pseudo)random number generator.
//	 */
//	static Genome random(in SpecimenParams params)
//	in
//	{
//		assert (&params);
//	}
//	out (result)
//	{
//		assert (result.hidden.length == params.layers + 1);
//		foreach (i, l; result.hidden)
//		{
//			if (i == result.hidden.length - 1)
//				assert (l.length == params.outputs);
//			else
//				assert (l.length == params.neurons);
//			
//			foreach (n; l)
//			{
//				if (i == 0)
//					assert (n.length == params.inputs + 1);
//				else
//					assert (n.length == params.neurons + 1);
//				
//				foreach (w; n)
//					assert (w >= params.weights.min && w <= params.weights.max);
//			}
//		}
//	}
//	body
//	{
//		Genome result;
//		
//		result.input = params.inputs;
//		
//		// Generate the first hidden layer
//		result.hidden ~= randomGenes(params.neurons, params.inputs, params.weights);
//		
//		// Generating remaining hidden layers
//		for (ulong i = 0; i < params.layers - 1; i++)
//			result.hidden ~= randomGenes(params.neurons, params.neurons, params.weights);
//		
//		// Output layer
//		result.hidden ~= randomGenes(params.outputs, params.neurons, params.weights);
//		
//		return result;
//	}
//	
//	// Force contract ckeck
//	unittest
//	{
//		import std.stdio;
//		
//		writeln("Genome.random(T)(in SpecimenParams params, ref T generator)");
//		
//		SpecimenParams sp;
//		sp.inputs  = 3;
//		sp.outputs = 2;
//		sp.layers  = 4;
//		sp.neurons = 5;
//		sp.weights.min = -10;
//		sp.weights.max =  10;
//		
//		Genome g = random(sp);
//	}
//	
//	/**
//	 * Crossover 2 genomes.
//	 *
//	 * Params:
//	 *     parents = A pair of parents' genomes to crossover.
//	 *     crossoverRate = Determines probability of gene exchange.
//	 *     alpha = Determines weigth of gene exchange. x1 = (1 - alpha) * y1 | x2 = alpha * y2
//	 *     generator = (Pseudo)random generator.
//	 *                 Produces randomnes to chose how genes will be crossed over.
//	 *
//	 * Returns:
//	 *     Array of exactly 2 genomes, which are results of crossing over 2 parant genomes.
//	 */
//	static Genome[2] crossover(in Genome[2] parents, in double crossoverRate, in double alpha)
//	in
//	{
//		assert (&parents[0]);
//		assert (&parents[1]);
//		
//		assert (parents[0].input         == parents[1].input        );
//		assert (parents[0].hidden.length == parents[1].hidden.length);
//		
//		foreach (li, layer; parents[0].hidden)
//		{
//			assert (layer.length == parents[1].hidden[li].length);
//			foreach (ni, neuron; layer)
//				assert (neuron.length == parents[1].hidden[li][ni].length);
//		}
//	}
//	out (result)
//	{
//		assert (&result[0]);
//		assert (&result[1]);
//	}
//	body
//	{
//		Genome[2] children;
//		
//		children[0].input = parents[0].input;
//		children[1].input = parents[1].input;
//		
//		children[0].hidden.length = parents[0].hidden.length;
//		children[1].hidden.length = parents[1].hidden.length;
//		
//		for (ulong li = 0; li < parents[0].hidden.length; li++)
//		{
//			children[0].hidden[li].length = parents[0].hidden[li].length;
//			children[1].hidden[li].length = parents[1].hidden[li].length;
//			
//			for (ulong ni = 0; ni < parents[0].hidden[li].length; ni++)
//			{
//				children[0].hidden[li][ni].length = parents[0].hidden[li][ni].length;
//				children[1].hidden[li][ni].length = parents[1].hidden[li][ni].length;
//				
//				for (ulong wi = 0; wi < parents[0].hidden[li][ni].length; wi++)
//				{
//					double roll   = uniform!"[)"(0.0, 1.0);
//					ulong  first  = (roll <  crossoverRate).to!ulong;
//					ulong  second = (roll >= crossoverRate).to!ulong;
//					
//					children[first ].hidden[li][ni][wi] = parents[0].hidden[li][ni][wi] *      alpha;
//					children[second].hidden[li][ni][wi] = parents[0].hidden[li][ni][wi] * (1 - alpha);
//				}
//			}
//		}
//		
//		return children;
//	}
//	
//	// Don't know how to properly test this...
//	unittest
//	{
//		import std.stdio : writeln;
//		
//		writeln("Genome.crossover(T)(in Genome[2] parents, ref T generator)");
//		
//		SpecimenParams sp;
//		sp.inputs  = 3;
//		sp.outputs = 2;
//		sp.layers  = 4;
//		sp.neurons = 5;
//		sp.weights.min = -10;
//		sp.weights.max =  10;
//		
//		Genome[2] p = [random(sp), random(sp)];
//		Genome[2] c = crossover(p, 0.5, 0.9);
//		
//		writeln(">>> Parent[0] = ", p[0]);
//		writeln(">>> Parent[1] = ", p[1]);
//		writeln(">>> Child [0] = ", c[0]);
//		writeln(">>> Child [1] = ", c[1]);
//	}
//	
//	/**
//	 * Mutates a genone.
//	 *
//	 * Randomly changes some genes, which are randomly selected too.
//	 *
//	 * Params:
//	 *     sp = Specimen parameters. Applies the same restriction to mutations like
//	 *          during genome generation.
//	 *     mutationRate = Determines how often genes will mutate.
//	 *     generator = (Pseudo)random generator. Produces radnomnes to decide
//	 *                 what genes and how they are going to mutate.
//	 */
//	void mutate(in SpecimenParams sp, in double mutationRate)
//	in
//	{
//		assert (&sp);
//		assert (&this);
//	}
//	body
//	{
//		foreach (layer; this.hidden)
//			foreach (neuron; layer)
//				foreach (ref weight; neuron)
//					if (uniform!"[)"(0.0, 1.0) < mutationRate)
//						weight = uniform!"[]"(sp.weights.min, sp.weights.max);
//	}
//	
//	// Don't know how to properly test this...
//	unittest
//	{
//		import std.stdio : writeln;
//		
//		writeln("Genome.mutate(T)(in SpecimenParams sp, ref T generator)");
//		
//		SpecimenParams sp;
//		sp.inputs  = 3;
//		sp.outputs = 2;
//		sp.layers  = 2;
//		sp.neurons = 3;
//		sp.weights.min = -10;
//		sp.weights.max =  10;
//		
//		Genome g = random(sp);
//		writeln("Before mutation = ", g);
//		g.mutate(sp, 0.05);
//		writeln("After mutation  = ", g);
//	}
}

