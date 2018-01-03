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
module layer;

import std.conv : to;

import neuron;

/**
 * Simple layer which only passes input values to a network internals.
 *
 * Does no calculations.
 */
struct InputLayer
{
	/**
	 * These neurons are just interface to a network.
	 */
	private InputNeuron[] neurons;
	
	/**
	 * Default constructor
	 *
	 * Params:
	 *     length = Number of neurons.
	 */
	this(in ulong length)
	{
		neurons.length = length;
	}
	
	/**
	 * Constructor with neuron initilization.
	 * 
	 * Generates value.length neurons.
	 *
	 * Params:
	 *     values = Array of input layers.
	 */
	this(in double[] values)
	{
		foreach (v; values)
			neurons ~= InputNeuron(v);
	}
	
	/**
	 * Examples:
	 * -------
	 * layer[2];
	 */
	double opIndex(ulong i)
	{
		return neurons[i]();
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opIndex(size_t i)");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i[2] == 6);
	}
	
	/**
	 * Examples:
	 * -------
	 * layer[2..5];
	 */
	double[] opSlice(ulong i, ulong j)
	{
		double[] result;
		for (ulong k = i; k < j; k++)
			result ~= neurons[k]();
		return result;
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opSlice(size_t i, size_t j)");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i[1..2] == [5]);
	}
	
	/**
	 * Examples:
	 * -------
	 * layer[0..$];
	 */
	ulong opDollar()
	{
		return this.length;
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opDollar()");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i[0..$] == [4, 5, 6]);
		assert (i[$-1]  ==  6);
	}
	
	/**
	 * Examples:
	 * -------
	 * double[] x = layer();
	 */
	double[] opCall()
	{
		return this[0..$];
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opCall()");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i() == [4, 5, 6]);
	}
	
	/**
	 * Examples:
	 * -------
	 * double[] x = layer([1, 2, 3]);
	 */
	double[] opCall(in double[] values)
	in
	{
		assert(values.length == neurons.length);
	}
	body
	{
		foreach(i, ref n; neurons)
			n(values[i]);
		return this[0..$];
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opCall(double[] values)");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i([1, 2, 3]) == [1, 2, 3]);
	}
	
	@property ulong length()
	{
		return neurons.length;
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.length()");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i.length == 3);
	}
	
	@property string toString(string indent = "")
	{
		string result = indent ~ "InputLayer:\n";
		foreach(i, n; neurons)
			result ~= n.toString(indent ~ "\t", i);
		return result;
	}
}

/**
 * Hidden layer.
 */
struct HiddenLayer
{
	private Neuron[] neurons;
	
	/**
	 * This flag determines whether sigmoid applies to results.
	 *
	 * It may be usefull if hidden layer is used as output layer.
	 */
	bool sig = true;
	
	this(double[][] chromosome)
	in
	{
		assert (chromosome.length     >= 1);
		assert (chromosome[0].length  >= 2); // 1 goes for bias
	}
	body
	{
		foreach(nGene; chromosome)
			neurons ~= Neuron(nGene[0..$-1], nGene[$-1]); // The last one is for bias
	}
	
	double opIndex(ulong i)
	{
		return neurons[i]();
	}
	
	double[] opSlice(ulong i, ulong j)
	{
		double[] result;
		foreach(n; neurons)
			result ~= n();
		return result[i..j];
	}
	
	double[] opCall()
	{
		double[] result;
		foreach(n; neurons)
			result ~= n();
		return result;
	}
	
	double[] opCall(T)(T prevLayer)
		if (is(T == InputLayer) || is(T == HiddenLayer))
	{
		double[] result;
		foreach(ref n; neurons)
			result ~= n(prevLayer[0 .. $], sig);
		return result;
	}
	
	ulong opDollar()
	{
		return this.neurons.length;
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
