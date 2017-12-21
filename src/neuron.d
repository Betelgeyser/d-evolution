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
module neuron;

import std.stdio     : writeln;
import std.random    : uniform;
import std.range     : generate, take;
import std.algorithm : each;
import std.conv      : to;
import std.math      : exp;
import std.array;

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

struct InputNeuron
{
	private double value;
	
	this(double value)
	{
		this.value = value;
	}
	
	double opCall()
	{
		return value;
	}
	
	double opCall(double value)
	{
		this.value = value;
		return this.value;
	}
	
	@property string toString(string indent = "", ulong num = 0)
	{
		string result = indent ~ "InputNeuron[" ~ num.to!string ~ "]:\n";
		result ~= indent ~ "\tValue = " ~ value.to!string ~ "\n";
		return result;
	}
}

unittest
{
	writeln("InputNeuron...");
	auto n = InputNeuron(3);
	assert (n()  == 3);
	assert (n(5) == 5);
}

struct RandomNeuron
{
	private
	{
		double   value;
		double[] weights;
		double   biasWeight;
	}
	
	this(T)(ulong inputLength, double minWeight, double maxWeight, ref T generator)
	{
		value       = 0;
		biasWeight  = uniform!"[]"(minWeight, maxWeight, generator);
		
		weights =
			generate(
				() => uniform!"[]"(minWeight, maxWeight, generator)
			).take(inputLength)
			.array();
	}
	
	double opIndex(size_t i)
	{
		return weights[i];
	}
	
	double[] opSlice(size_t i, size_t j)
	{
		return weights[i..j];
	}
	
	double opCall()
	{
		return value;
	}
	
	double opCall(double[] inputs, bool sig = true)
	{
		assert(inputs.length == weights.length);
		
		value = 0;
		
		foreach(i, w; weights)
			value += w * inputs[i];
		
		value += biasWeight;
		
		if (sig)
			value = sigmoid(value);
		
		return value;
	}
	
	@property size_t length()
	{
		return weights.length;
	}
	
	@property string toString(string indent = "", ulong num = 0)
	{
		string result = indent ~ "RandomNeuron[" ~ num.to!string ~ "]:\n";
		result ~= indent ~ "\tValue = " ~ value.to!string ~ "\n";
		result ~= indent ~ "\tBias weight = " ~ biasWeight.to!string ~ "\n";
		result ~= indent ~ "\tWeights:\n";
		
		weights.each!(
			(i, w) =>
				result ~= indent ~ "\t\tWeight[" ~ i.to!string ~ "] = "~ w.to!string ~ "\n"
		)();
		
		return result;
	}
}

unittest
{
	writeln("RandomNeuron...");
	import std.random : Mt19937_64;
	auto rng = Mt19937_64(0);
	
	auto n = RandomNeuron(5, -10, 10, rng);
	
	assert (n()      == 0);
	assert (n.length == 5);
	
	foreach (w; n[0 .. n.length - 1])
	{
		assert (w <=  10);
		assert (w >= -10);
	}
}
