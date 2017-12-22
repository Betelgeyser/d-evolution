/**
 * Copyright © 2017 Sergei Iurevich Filippov, All Rights Reserved.
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

import std.random : uniform;
import std.range  : generate, take;
import std.conv   : to;
import std.array;

/** Min and max weight value of a neuron connection. */
struct WeightInterval
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
	
	WeightInterval weights; /// Min and max weight value of a neuron connection.
	
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
 * Genetare random neuron weights.
 */
double[] generateNeuronGenes(T)(in ulong weightNumber, in WeightInterval weight, ref T generator)
in
{
	assert (weightNumber >= 1);
	assert (&weight);
}
out (result)
{
	assert (result.length == weightNumber + 1); // +1 goes for bias
	foreach (w; result)
		assert (w >= weight.min && w <= weight.max);
}
body
{
	return generate(
		() => uniform!"[]"(weight.min, weight.max, generator)
	).take(weightNumber + 1) // +1 goes for bias
	 .array;
}

// Force contract call
unittest
{
	import std.random : Mt19937_64, unpredictableSeed;
	
	auto rng = Mt19937_64(unpredictableSeed());
	
	WeightInterval wi;
	wi.min = -10;
	wi.max =  10;
	
	double[] g = generateNeuronGenes(5, wi, rng);
}

struct InputChromosome
{
	ulong gene;
	alias gene this;
	
	this(in SpecimenParams params)
	in
	{
		assert (&params);
	}
	body
	{
		gene = params.inputs;
	}
}

struct HiddenChromosome
{
	double[][][] genes; // The first index is a layer, the second one is a neuron and the third one is a connection weight.
	alias genes this;
	
	this(T)(in SpecimenParams params, ref T generator)
	in
	{
		assert (&params);
	}
	out
	{
		assert (genes.length == params.layers);
		foreach (i, l; genes)
		{
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
		// Generate the first hidden layer
		double[][] tmp_1;
		for (long i = 0; i < params.neurons; i++)
			tmp_1 ~= generateNeuronGenes(params.inputs, params.weights, generator);
		
		genes ~= tmp_1;
		
		// Generating remaining hidden layers
		for (ulong i = 0; i < params.layers - 1; i++)
		{
			double[][] tmp_2;
			for (ulong j = 0; j < params.neurons; j++)
				tmp_2 ~= generateNeuronGenes(params.neurons, params.weights, generator);
			
			genes ~= tmp_2;
		}
	}
}

struct OutputChromosome
{
	double[][] genes; // The first index is a neuron, the second one is a connection weight
	alias genes this;
	
	this(T)(in SpecimenParams params, ref T generator)
	in
	{
		assert (&params);
	}
	out
	{
		assert (genes.length == params.outputs);
		
		foreach (n; genes)
		{
			assert (n.length == params.neurons + 1);
			foreach (w; n)
				assert (w >= params.weights.min && w <= params.weights.max);
		}
	}
	body
	{
		double[][] tmp;
		for (long i = 0; i < params.outputs; i++)
			tmp ~= generateNeuronGenes(params.neurons, params.weights, generator);
		
		genes = tmp;
	}
}
	
// Force contracts call HiddenChromosome, OutputChromosome
unittest
{
	import std.random : Mt19937_64, unpredictableSeed;
	
	auto rng = Mt19937_64(unpredictableSeed());
	
	SpecimenParams sp;
	sp.inputs  = 3;
	sp.outputs = 2;
	sp.layers  = 4;
	sp.neurons = 5;
	sp.weights.min = -10;
	sp.weights.max =  10;
	
	HiddenChromosome h = HiddenChromosome(sp, rng);
	OutputChromosome o = OutputChromosome(sp, rng);
}

struct Genome
{
	InputChromosome  input;
	HiddenChromosome hidden;
	OutputChromosome output;
	
	private
	{
		static immutable double crossoverProbability = 0.5; /// Determines probability of gene exchange.
		static immutable double alpha                = 1.9; /// Determines weigth of gene exchange. x1 = (1 - alpha) * y1 | x2 = alpha * y2
	}
	
	invariant
	{
		assert (crossoverProbability >= 0.0 && crossoverProbability <= 1.0);
		assert (alpha                >= 0.0 && alpha                <= 1.0);
	}
	
	/**
	 * Generate random genome.
	 *
	 * Params:
	 *     params = Parameters of generated network specimen.
	 *     generator = (Pseudo)random number generator.
	 */
	static Genome generateRandom(T)(in SpecimenParams params, ref T generator)
	in
	{
		assert (&params);
	}
	body
	{
		Genome result;
		
		result.input  = InputChromosome (params);
		result.hidden = HiddenChromosome(params, generator);
		result.output = OutputChromosome(params, generator);
		
		return result;
	}
	
//	/**
//	 * Crossover 2 genomes.
//	 *
//	 */
//	static Genome[2] crossover(T)(in Genome[2] g, ref T generator)
//	{		
//		scope double[] probabilities = generate(
//			() => uniform!"[)"(0.0, 1.0, generator)
//		).take(parents[0].length)
//		 .array;
//		
//		double[][] children;
//		foreach (i, p; probabilities)
//		{
//			children[(p <  crossoverProbability).to!ulong] ~= parents[0][i] *      alpha;
//			children[(p >= crossoverProbability).to!ulong] ~= parents[1][i] * (1 - alpha);
//		}
//		
//		return children;
//	}
//	
//	// Don't know how to properly test this...
//	unittest
//	{
//		import std.stdio : writeln;
//		import std.random : Mt19937_64, unpredictableSeed;
//		
//		writeln("Genome[2] crossover(T)(in Genome[2] g, in ulong splitPoint, in double alpha, ref T generator)");
//		
//		auto rng = Mt19937_64(unpredictableSeed());
//		
//		SpecimenParams sp;
//		sp.inputs  = 3;
//		sp.outputs = 2;
//		sp.layers  = 4;
//		sp.neurons = 5;
//		sp.minWeight = -10;
//		sp.maxWeight =  10;
//		
//		Genome[2] p = [generateRandom(sp, rng), generateRandom(sp, rng)];
//		Genome[2] c = crossover(p, rng);
//		
//		writeln("Parent 1:", p[0]);
//		writeln("Parent 2:", p[1]);
//		writeln("Child 1:",  c[0]);
//		writeln("Child 2:",  c[1]);
//	}
}

