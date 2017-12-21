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
module layer;

import std.random : uniform;
import std.range  : generate, take;
import std.conv   : to;
import std.array;

import neuron;

struct InputLayer
{
	private InputNeuron[] neurons;
	
	this(ulong size)
	{
		neurons.length = size;
	}
	
	this(double[] values)
	{
		foreach (v; values)
			neurons ~= InputNeuron(v);
	}
	
	double opIndex(size_t i)
	{
		return neurons[i]();
	}
	
	double[] opSlice(size_t i, size_t j)
	{
		double[] result;
		for (size_t k = i; k <= j; k++)
			result ~= neurons[k]();
		return result;
	}
	
	double[] opCall()
	{
		return this[0 .. this.length - 1];
	}
	
	double[] opCall(double[] values)
	{
		assert(values.length == neurons.length);
		foreach(i, ref n; neurons)
			n(values[i]);
		return this[0 .. this.length - 1];
	}
	
	@property size_t length()
	{
		return neurons.length;
	}
	
	@property string toString(string indent = "")
	{
		string result = indent ~ "InputLayer:\n";
		foreach(i, n; neurons)
			result ~= n.toString(indent ~ "\t", i);
		return result;
	}
}

unittest
{
	import std.stdio : writeln;
	writeln("InputLayer...");
	
	auto i1 = InputLayer(6);
	assert(i1.length == 6);
	
	i1([1, 2, 3, 4, 5, 6]);
	assert (i1[0 .. i1.length - 1] == [1, 2, 3, 4, 5, 6]);
	
	auto i2 = InputLayer([1, 2, 3, 4, 5]);
	assert (i2()                   == [1, 2, 3, 4, 5]);
	assert (i2[0 .. i2.length - 1] == [1, 2, 3, 4, 5]);
}

struct HiddenLayer
{
	private RandomNeuron[] neurons;
	bool sig = true;
	
	this(T)(ulong size, ulong prevSize, double minWeight, double maxWeigth, ref T generator)
	in
	{
		assert (size     >= 1, "Layer must have at least 1 neuron.");
		assert (prevSize >= 1, "Layer must have at least 1 neuron.");
		
		assert (maxWeigth >= minWeight, "Max neuron weight must be greater or equal than min weight.");
	}
	body
	{
		neurons = generate(
			() => RandomNeuron(prevSize, minWeight, maxWeigth, generator)
		).take(size)
		.array();
	}
	
	double opIndex(size_t i)
	{
		return neurons[i]();
	}
	
	double[] opSlice(size_t i, size_t j)
	{
		double[] result;
		foreach(n; neurons)
			result ~= n();
		return result;
	}
	
	double[] opCall()
	{
		double[] result;
		foreach(n; neurons)
			result ~= n();
		return result;
	}
	
	double[] opCall(T)(T prevLayer)
	{
		double[] result;
		foreach(ref n; neurons)
			result ~= n(prevLayer[0 .. prevLayer.length - 1], sig);
		return result;
	}
	
	@property size_t length()
	{
		return neurons.length;
	}
	
	@property string toString(string indent = "", ulong num = 0)
	{
		string result = indent ~ "HiddenLayer[" ~ num.to!string ~ "]:\n";
		foreach(i, n; neurons)
			result ~= n.toString(indent ~ "\t", i);
		return result;
	}
}
