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

/**
 * Random genome generation parameters.
 */
struct SpecimenParams
{
	ulong inputs;  /// Number of network's inputs.
	ulong outputs; /// Number of network's outputs.
	ulong layers;  /// Number of hidden layers.
	ulong neurons; /// Number of neurons in every hidden layer.
	
	/** Min and max weight value of a neuron connection. */
	double minWeight;
	double maxWeight;
	
	invariant
	{
		assert (inputs  >= 1);
		assert (outputs >= 1);
		assert (layers  >= 1);
		assert (neurons >= 1);
		
		assert (minWeight <= maxWeight);
	}
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
		assert (genes      .length == params.layers    );
		assert (genes[0]   .length == params.neurons   );
		assert (genes[0][0].length == params.inputs + 1);
		
		if (genes.length >= 2)
			assert (genes[1][0].length == params.neurons + 1);
		
		foreach (l; genes)
			foreach (n; l)
				foreach (w; n)
					assert (w >= params.minWeight && w <= params.maxWeight);
	}
	body
	{
		// Generate the first hidden layer
		double[][] tmp_1;
		for (long i = 0; i < params.neurons; i++)
		{
			tmp_1 ~= generate(
				() => uniform!"[]"(params.minWeight, params.maxWeight, generator)
			).take(params.inputs + 1) // +1 goes for bias
			 .array;
		}
		genes ~= tmp_1;
		
		// Generating remaining hidden layers
		for (ulong i = 0; i < params.layers - 1; i++)
		{
			double[][] tmp_2;
			for (ulong j = 0; j < params.neurons; j++)
			{
				tmp_2 ~= generate(
					() => uniform!"[]"(params.minWeight, params.maxWeight, generator)
				).take(params.neurons + 1) // +1 goes for bias
				 .array;
			}
			genes ~= tmp_2;
		}
	}
}

struct OutputChromosome
{
	double[][] genes;
	alias genes this;
	
	this(T)(in SpecimenParams params, ref T generator)
	in
	{
		assert (&params);
	}
	out
	{
		assert (genes   .length == params.outputs    );
		assert (genes[0].length == params.outputs + 1);
		
		foreach (n; genes)
			foreach (w; n)
				assert (w >= params.minWeight && w <= params.maxWeight);
	}
	body
	{
		double[][] tmp;
		for (long i = 0; i < params.outputs; i++)
		{
			tmp ~= generate(
				() => uniform!"[]"(params.minWeight, params.maxWeight, generator)
			).take(params.neurons + 1) // +1 goes for bias
			 .array;
		}
		genes = tmp;
	}
}

struct Genome
{
	InputChromosome  input;
	HiddenChromosome hidden;
	double[][]   output;
	
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
		
		result.input  = InputChromosome(params);
		result.hidden = HiddenChromosome(params, generator);
		result.output = OutputChromosome(params, generator);
		
		return result;
	}

	unittest
	{
		import std.stdio : writeln;
		import std.random : Mt19937_64, unpredictableSeed;
		
		writeln("Genome.generate(T)(in SpecimenParams params, ref T generator)");
		
		auto rng = Mt19937_64(unpredictableSeed());
		
		SpecimenParams sp;
		sp.inputs  = 3;
		sp.outputs = 2;
		sp.layers  = 4;
		sp.neurons = 5;
		sp.minWeight = -10;
		sp.maxWeight =  10;
		
		Genome g = Genome.generateRandom(sp, rng);
		assert (g.input               == 3    );
		assert (g.output      .length == 2    );
		assert (g.output[0]   .length == 5 + 1);
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

