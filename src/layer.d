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

// Standard D modules
import std.conv      : to;
import std.algorithm : map;
import std.array;

// dnn
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
	private double[] neurons;
	
	/**
	 * Default constructor
	 *
	 * Params:
	 *     length = Number of neurons.
	 */
	this(in ulong length) pure nothrow @safe
	{
		neurons.length = length;
	}
	
	/**
	 * Constructor with neuron initilization.
	 * 
	 * Generates `value.length` neurons.
	 *
	 * Params:
	 *     values = Array of input layers.
	 */
	this(in double[] values) pure nothrow @safe
	{
		foreach (v; values)
			neurons ~= v;
	}
	
	double opIndex(size_t i) const pure nothrow @safe @nogc
	{
		return neurons[i];
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opIndex(size_t i)");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i[2] == 6);
	}
	
	const(double[]) opSlice(size_t i, size_t j) const pure nothrow @safe @nogc
	{
		return neurons[i..j];
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opSlice(size_t i, size_t j)");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i[1..2] == [5]);
	}
	
	size_t opDollar() const pure nothrow @safe @nogc
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
	 * Rerurns all neurons values.
	 *
	 * Note: does NOT evaluate new values.
	 */
	const(double[]) opCall() const pure nothrow @safe @nogc
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
	 * Evaluate neurons' values.
	 *
	 * Params:
	 *     values = Values to work on. Basically are outputs of a previous layer.
	 *
	 * Returns:
	 *     New neurons values.
	 */
	const(double[]) opCall(in double[] values) pure nothrow @safe @nogc
	in
	{
		assert(values.length == neurons.length);
	}
	body
	{
		foreach(i, ref n; neurons)
			n = values[i];
		return this[0..$];
	}
	
	unittest
	{
		import std.stdio : writeln;
		writeln("InputLayer.opCall(double[] values)");
		InputLayer i = InputLayer([4, 5, 6]);
		assert (i([1, 2, 3]) == [1, 2, 3]);
	}
	
	/**
	 * Returns neurons number.
	 */
	@property size_t length() const pure nothrow @safe @nogc
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
	
	/**
	 * Neuron's human-readable string representation.
	 */
	@property string toString(string indent = "") const @safe
	{
		string result = indent ~ "InputLayer:\n";
		foreach(i, n; neurons)
			result ~= indent ~ "InputNeuron[" ~ i.to!string ~ "]:\n"
				~ indent ~ "\tValue = " ~ n.to!string ~ "\n";
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
	
	this(in double[][] chromosome) pure nothrow @safe
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
	
	double opIndex(in size_t i) const pure nothrow @safe @nogc
	{
		return neurons[i]();
	}
	
	double[] opSlice(in size_t i, in size_t j) const pure nothrow @safe
	{
		double[] result;
		for (size_t k = 0; k < j; k++)
			result ~= neurons[k]();
		return result;
	}
	
	double[] opCall() const pure nothrow @safe
	{
		return this[0..$];
	}
	
	double[] opCall(T)(in T prevLayer) pure nothrow @safe
		if (is(T == InputLayer) || is(T == HiddenLayer))
	{
		double[] result;
		foreach(ref n; neurons)
			result ~= n(prevLayer[0..$], sig);
		return result;
	}
	
	size_t opDollar() const pure nothrow @safe @nogc
	{
		return this.neurons.length;
	}
	
	/**
	 * Returns neurons number.
	 */
	@property size_t length() const pure nothrow @safe @nogc
	{
		return neurons.length;
	}
	
	/**
	 * Neuron's human-readable string representation.
	 */
	@property string toString(in string indent = "", in ulong num = 0) const @safe
	{
		string result = indent ~ "HiddenLayer[" ~ num.to!string ~ "]:\n";
		foreach(i, n; neurons)
			result ~= n.toString(indent ~ "\t", i);
		return result;
	}
}
