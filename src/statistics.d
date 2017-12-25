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
module statistics;

import std.algorithm : sum, map;
import std.math      : pow, abs;

/**
 * Absolute error between real and approximated value.
 *
 * Params:
 *     xTrue = Real value.
 *     xApprox = Approximation of the same value.
 */
double absoluteError(double xTrue, double xApprox)
{
	return abs(xApprox - xTrue);
}

unittest
{
	import std.math : approxEqual;
	
	assert (approxEqual(
			absoluteError(10_000_000.0, 10_000_000.1),
			0.1,
			0.000_001
		));
	
	assert (approxEqual(
			absoluteError(0.000_000_000_1, 0.000_000_000_101),
			0.000_000_000_001,
			0.000_000_000_000_01
		));
}

/**
 * Relative error between real and approximated value.
 *
 * Params:
 *     xTrue = Real value.
 *     xApprox = Approximation of the same value. 
 */
double relativeError(double xTrue, double xApprox)
{
	return absoluteError(xTrue, xApprox) / xTrue;
}

unittest
{
	import std.math : approxEqual;
	
	assert (approxEqual(
			relativeError(10_000_000, 10_000_001),
			0.000_000_1,
			0.000_000_001
		));
	
	assert (approxEqual(
			relativeError(0.000_000_000_1, 0.000_000_000_101),
			0.01,
			0.000_001
		));
}

/**
 * Mean value of a given sample.
 *
 * Params:
 *     sample = Data sample.   
 */
double mean(double[] sample)
{
	return sample.map!(x => x / sample.length).sum;
}

unittest
{
	import std.math : approxEqual;
	
	assert (approxEqual(
			mean([1_000_000_000, 1_000_000_001, 999_999_999]),
			1_000_000_000,
			0.000_001
		));
	
	assert (approxEqual(
			mean([0.000_000_1, 0.000_000_99, 0.000_000_11]),
			0.000_000_1,
			0.000_000_001
		));
}

/**
 * Standard error of a given sample.
 *
 * Params:
 *     sample = Data sample.
 */
double standardError(double[] sample)
{
	return pow(sample.map!(x => pow(x - mean(sample), 2)).sum / sample.length, 0.5);
}
	
unittest
{
	import std.math : approxEqual;
	
	assert (approxEqual(
			standardError([1_000_000_000, 1_000_000_001, 999_999_999]),
			0.8165
		));
	
	assert (approxEqual(
			standardError([0.000_000_000_1, 0.000_000_000_11, 0.000_000_000_09]),
			0.000_000_000_008_165,
			0.000_000_000_000_001
		));
}

