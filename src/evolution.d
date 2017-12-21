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
}

struct Genome
{
	ulong input;
	double[][][] hidden;
	double[][] output;
	
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
		with (params)
		{
			assert (inputs  >= 1);
			assert (outputs >= 1);
			assert (layers  >= 1);
			assert (neurons >= 1);
			assert (minWeight <= maxWeight);
		}
	}
	body
	{
		Genome result;
		
		// Input layer
		result.input = params.inputs;
		
		// Generate the first hidden layer
		double[][] tmp_1;
		for (long i = 0; i < params.neurons; i++)
		{
			tmp_1 ~= generate(
				() => uniform!"[]"(params.minWeight, params.maxWeight, generator)
			).take(params.inputs + 1) // +1 goes for bias
			 .array;
		}
		result.hidden ~= tmp_1;
		
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
			result.hidden ~= tmp_2;
		}
		
		// Output layer
		double[][] tmp_3;
		for (long i = 0; i < params.outputs; i++)
		{
			tmp_3 ~= generate(
				() => uniform!"[]"(params.minWeight, params.maxWeight, generator)
			).take(params.neurons + 1) // +1 goes for bias
			 .array;
		}
		result.output ~= tmp_3;
		
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
		sp.maxWeight = -10;
		
		Genome g = Genome.generateRandom(sp, rng);
		assert (g.input               == 3    );
		assert (g.output      .length == 2    );
		assert (g.output[0]   .length == 5 + 1);
		assert (g.hidden      .length == 4    );
		assert (g.hidden[0]   .length == 5    );
		assert (g.hidden[0][0].length == 3 + 1);
		assert (g.hidden[1][0].length == 5 + 1);
	}
}

