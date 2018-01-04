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
	
	/**
	 * Default constructor.
	 *
	 * Params:
	 *     weights = Neuron input's weights.
	 *     bias = Bias constant.
	 */
	this(in double[] weights, in double bias) pure nothrow @safe
	{
		this.value   = 0;
		this.bias    = bias;
		this.weights = weights.idup;
	}
	 
	double opIndex(in size_t i) const pure nothrow @safe @nogc
	{
		return weights[i];
	}
	
	const(double[]) opSlice(in size_t i, in size_t j) const pure nothrow @safe
	{
		return weights[i..j];
	}
	
	/**
	 * Return current neuron value.
	 *
	 * Note:
	 *     Returns curren neuron value and does NOT calculate it based on inputs.
	 */
	double opCall() const pure nothrow @safe @nogc
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
	double opCall(in double[] inputs, in bool sig = true) pure nothrow @safe @nogc
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
	
	/**
	 * Neuron's length which is number of neuron's inputs.
	 */
	@property size_t length() const pure nothrow @safe @nogc
	{
		return weights.length;
	}
	
	/**
	 * Neuron's human-readable string representation.
	 */
	@property string toString(in string indent = "", in ulong num = 0) const @safe
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
