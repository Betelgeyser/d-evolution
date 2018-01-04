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
module neuron;

// Standard D modules
import std.algorithm : each;
import std.conv      : to;

// dnn
import statistics : sigmoid;

/**
 * Simple neuron.
 *
 * Does no calculations, just propogates its value to hidden neurons.
 */
struct InputNeuron
{
	/**
	 * Current neuron value.
	 */
	private double value;
	
	this(double value)
	{
		this.value = value;
	}
	
	/**
	 * Returns current neuron value.
	 */
	double opCall()
	{
		return value;
	}
	
	/**
	 * Sets neuron value and returns it.
	 */
	double opCall(double value)
	{
		this.value = value;
		return this.value;
	}

	unittest
	{
		import std.stdio : writeln;
		writeln("InputNeuron.opCall(double value)");
		auto n = InputNeuron(3);
		assert (n()  == 3);
		assert (n(5) == 5);
	}
	
	/**
	 * Neuron's human-readable string representation.
	 */
	@property string toString(string indent = "", ulong num = 0)
	{
		string result = indent ~ "InputNeuron[" ~ num.to!string ~ "]:\n";
		result ~= indent ~ "\tValue = " ~ value.to!string ~ "\n";
		return result;
	}
}

/**
 * Basic neuron with bias.
 */
struct Neuron
{
	private
	{
		immutable double[] weights; /// Neuron's weigths.
		immutable double   bias;    /// Bias constant.
		
		double value; /// Current neuron's value.
	}
	
	this(in double[] weights, in double bias)
	{
		this.value   = 0;
		this.bias    = bias;
		this.weights = weights.idup;
	}
	
	double opIndex(ulong i)
	{
		return weights[i];
	}
	
	double[] opSlice(ulong i, ulong j)
	{
		return weights[i..j].dup;
	}
	
	/**
	 * Reruen current neuron value.
	 *
	 * Note:
	 *     Neuron.opCall() returns curren neuron value and does NOT calculate it based on inputs.
	 */
	double opCall()
	{
		return value;
	}
	
	/**
	 * Calculate neuron value from a given input.
	 *
	 * Params:
	 *     inputs = Array of input data.
	 *     sig = If set to `true` will applay sigmoid function to the result.
	 */ 
	double opCall(double[] inputs, bool sig = true)
	in
	{
		assert(inputs.length == weights.length);
	}
	body
	{
		value = 0;
		
		foreach(i, w; weights)
			value += w * inputs[i];
		
		value += bias;
		
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
		string result = indent ~ "Neuron[" ~ num.to!string ~ "]:\n";
		result ~= indent ~ "\tValue = " ~ value.to!string ~ "\n";
		result ~= indent ~ "\tBias weight = " ~ bias.to!string ~ "\n";
		result ~= indent ~ "\tWeights:\n";
		
		weights.each!(
			(i, w) =>
				result ~= indent ~ "\t\tWeight[" ~ i.to!string ~ "] = "~ w.to!string ~ "\n"
		)();
		
		return result;
	}
}
