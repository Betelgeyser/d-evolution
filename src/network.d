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

import layer;
	import std.stdio;

struct Network
{
	InputLayer  inputLayer;  /// Input layer.
	HiddenLayer outputLayer; /// Output layer.
	
	private HiddenLayer[] hiddenLayers;
	
	this(T)(ulong inputs, ulong outputs, ulong lNumber, ulong nNumber, double minWeight, double maxWeigth, ref T generator)
	in
	{
		assert (inputs  >= 1, "Network must have at least 1 input neuron.");
		assert (outputs >= 1, "Network must have at least 1 output neuron.");
		assert (lNumber >= 1, "Network must have at least 1 hidden layer.");
		assert (nNumber >= 1, "Each of network's hidden layers must have at least 1 neuron.");
		
		assert (maxWeigth >= minWeight, "Max neuron weight must be greater or equal than min weight.");
	}
	body
	{
		inputLayer = InputLayer(inputs);
		
		hiddenLayers ~= HiddenLayer(nNumber, inputLayer.length, minWeight, maxWeigth, generator);
		
		for(ulong i = 1; i < lNumber; i++)
			hiddenLayers ~= HiddenLayer(nNumber, hiddenLayers[i - 1].length, minWeight, maxWeigth, generator);
		
		outputLayer = HiddenLayer(outputs, hiddenLayers[hiddenLayers.length - 1].length, minWeight, maxWeigth, generator);
		outputLayer.sig = false;
	}
	
	double[] opCall()
	{
		return outputLayer();
	}
	
	double[] opCall(double[] input)
	{
		assert (input.length == inputLayer.length);
		
		inputLayer(input);
		hiddenLayers[0](inputLayer);
		
		for (ulong i = 1; i < hiddenLayers.length; i++)
			hiddenLayers[i](hiddenLayers[i - 1]);
		
		outputLayer(hiddenLayers[hiddenLayers.length - 1]);
		
		return outputLayer();
	}
	
	@property string toString()
	{
		string result = "RandomNetwork:\n";
		result ~= inputLayer.toString("\t");
		foreach(i, h; hiddenLayers)
			result ~= h.toString("\t", i);
		result ~= outputLayer.toString("\t");
		return result;
	}
}

unittest
{
	import std.stdio;
	writeln("RandomNetwork...");
	
	import std.random : Mt19937_64;
	auto rng = Mt19937_64(0);
	
	auto rn = Network(5, 3, 2, 5, -10, 10, rng);
	assert (rn() == [0, 0, 0]);
}