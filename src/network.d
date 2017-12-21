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
module network;

import evolution : Genome;
import layer;

/**
 * Simple feedforward network.
 */
struct Network
{
	/**
	 * Input layer.
	 *
	 * Contains neurons that are simply passes input values to hidden layers.
	 * Number of neurons must be equal to a number of input values.
	 */
	InputLayer  inputLayer;
	
	/**
	 * Output layer.
	 *
	 * Contains neurons that produces output values visible from an outside of a network.
	 * Number of neurons must be equal to a number of input values.
	 */
	HiddenLayer outputLayer;
	
	/**
	 * Hidden layers.
	 *
	 * Contains pure magic. Each layer propogates its values to a next layer and finelly to the output layer.
	 */ 
	private HiddenLayer[] hiddenLayers;
	
	/**
	 * Spawns a network from a giver genetic material.
	 *
	 * Params:
	 *     genome = Genetic material of the network consisting of chomosomes.
	 *              The first cromosome must contain an encoded input layer.
	 *              The second one must contain encoded hidden layers.
	 *              And the last one must contain an encoded output layer.
	 *              Such chomosome structure is essential to propper crossing over and mutating.
	 */
	this(Genome genome)
	{
		inputLayer = InputLayer(genome.input);
		
		foreach(lGene; genome.hidden)
		{
			ulong neurons = lGene.length;
			ulong weights = lGene[0].length;
			
			hiddenLayers ~= HiddenLayer(lGene);
		}
		
		outputLayer = HiddenLayer(genome.output);
		outputLayer.sig = false;
	}
	
	/**
	 * Return network's outputs.
	 *
	 * Note:
	 *     Network.opCall() does NOT reevaluate outputs, it just returns values of the output layer.
	 */
	double[] opCall()
	{
		return outputLayer();
	}
	
	/**
	 * Return network's outputs.
	 *
	 * Params:
	 *     input = Input values to work on.
	 */
	double[] opCall(double[] input)
	in
	{
		assert (input.length == inputLayer.length);
	}
	body
	{		
		inputLayer(input);
		
		foreach (i, ref h; hiddenLayers)
			if (i == 0)
				h(inputLayer);
			else
				h(hiddenLayers[i - 1]);
		
		outputLayer(hiddenLayers[$ - 1]);
		
		return outputLayer();
	}
	
	unittest
	{
	import std.stdio : writeln;
		writeln("Network");
		Genome g;
		
		g.input = 2;
		
		g.hidden = [
			[ [1, 2, 3   ], [3, 2, 1   ], [1, 0, 1] ],
			[ [1, 1, 1, 1], [2, 2, 2, 2]         ]
		];
		
		g.output = [ [2, 1, 2] ];
		
		Network n = Network(g);
		assert (n.length             == 2);
		assert (n.inputLayer.length  == 2);
		assert (n.outputLayer.length == 1);
		assert (n()                  == [0]);
		
		n([0, 0]);
	}
	
	/**
	 * Return hidden layers number.
	 *
	 * Note:
	 *     Input and output layers do not count.
	 */
	@property ulong length()
	{
		return hiddenLayers.length;
	}
	
	@property string toString()
	{
		string result = "Network:\n";
		result ~= inputLayer.toString("\t");
		foreach(i, h; hiddenLayers)
			result ~= h.toString("\t", i);
		result ~= outputLayer.toString("\t");
		return result;
	}
}

