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
module dnn.math;

import std.algorithm : sum, map;
import std.math      : pow, abs, sqrt, exp;

/**
 * Sigmoid function.
 */
double sigmoid(double x) pure nothrow @safe @nogc
{
	return 1 / (1 + exp(-x));
}

/**
 * Vector magnitude in the Euclidean vector space.
 *
 * Params:
 *     vector = Vector.
 */
double magnitude(in double[] vector) pure nothrow @safe @nogc // Pop, pop!
{
	return sqrt(
		vector.map!"pow(a, 2)".sum
	);
}

unittest
{
	import std.stdio : writeln;
	import std.math  : approxEqual;
	
	writeln("statistics.magnitude(double[] vector)");
	
	assert (approxEqual(
			magnitude( [100] ),
			100,
			0.000_001
		));
	
	assert (approxEqual(
			magnitude( [0, 0, 0, 0] ),
			0,
			0.000_001
		));
	
	assert (approxEqual(
			magnitude( [4, 3] ),
			5,
			0.000_001
		));
}

/**
 * Absolute error.
 *
 * Params:
 *     vTrue = Actual vector.
 *     vApprox = Approximated vector.
 *
 * Returns:
 *     Absolute error between actual and approximated vectors
 */
double AE(in double[] vTrue, in double[] vApprox) pure nothrow @safe
{
	assert (vTrue.length == vApprox.length);
	
	double[] diff;
	diff.length = vTrue.length;
	diff[] = vTrue[] - vApprox[];
	
	return magnitude(diff);
}

unittest
{
	import std.stdio : writeln;
	import std.math  : approxEqual;
	
	writeln("statistics.AE(in double[] vTrue, in double[] vApprox)");
	
	assert (approxEqual(
			AE([10_000_000.0], [10_000_000.1]),
			0.1,
			0.000_001
		));
	
	assert (approxEqual(
			AE([0.000_000_000_100], [0.000_000_000_101]),
			0.000_000_000_001_00,
			0.000_000_000_000_01
		));
	
	assert (approxEqual(
			AE( [1.0, 1.0], [2.0, 2.0] ),
			1.414_21,
			0.000_01
		));
}

/**
 * Relative error.
 *
 * Params:
 *     vTrue = Actual vector.
 *     vApprox = Approximated vector.
 *
 * Returns:
 *     Relative error between actual and approximated vectors. 
 */
double RE(in double[] vTrue, in double[] vApprox) pure nothrow @safe
{
	return AE(vTrue, vApprox) / magnitude(vTrue);
}

unittest
{
	import std.stdio : writeln;
	import std.math : approxEqual;
	
	writeln("statistics.RE(in double[] vTrue, in double[] vApprox)");
	
	assert (approxEqual(
			RE([10_000_000], [10_000_001]),
			0.000_000_100,
			0.000_000_001
		));
	
	assert (approxEqual(
			RE([0.000_000_000_1], [0.000_000_000_101]),
			0.01,
			0.000_001
		));
	
	assert (approxEqual(
			RE( [3, 4], [3.000_001, 3.999_999] ),
			0.000_000_283,
			0.000_000_001
		));
}

/**
 * Mean absolute relative error.
 *
 * Params:
 *     sTrue = Sample of real data.
 *     sApprox = Sample of approximated data.
 *
 * Returns;
 *     Mean absolute relative error between given data samples.
 */
double MARE(in double[][] sTrue, in double[][] sApprox) pure nothrow @safe
in
{
	assert (sTrue.length == sApprox.length);
	foreach (i, v; sTrue)
		assert (v.length == sApprox[i].length);
}
body
{
	scope double[] tmp;
	for (ulong i = 0; i < sTrue.length; i++)
		tmp ~= RE(sTrue[i], sApprox[i]);
	
	return tmp.sum / sTrue.length;
}

unittest
{
	import std.stdio : writeln;
	import std.math : approxEqual;
	
	writeln("statistics.MARE(in double[][] sTrue, in double[][] sApprox)");
	
	assert (approxEqual(
			MARE(
				[ [1_000_000_000.0], [-2_000_000_000.0], [3_000_000_000.0] ],
				[ [1_000_000_000.0], [-2_000_000_001.0], [2_999_999_999.0] ]
			),
			0.000_000_000_278,
			0.000_000_000_001
		));
	
	assert (approxEqual(
			MARE(
				[ [0.000_000_10], [0.000_000_20], [-0.000_000_30] ],
				[ [0.000_000_11], [0.000_000_19], [-0.000_000_30] ]
			),
			0.050_000,
			0.000_001
		));
}

///**
// * Standard error of a given sample.
// *
// * Params:
// *     sample = Data sample.
// */
//double standardError(double[] sample)
//{
//	return pow(sample.map!(x => pow(x - mean(sample), 2)).sum / sample.length, 0.5);
//}
//	
//unittest
//{
//	import std.math : approxEqual;
//	
//	assert (approxEqual(
//			standardError([1_000_000_000, 1_000_000_001, 999_999_999]),
//			0.8165
//		));
//	
//	assert (approxEqual(
//			standardError([0.000_000_000_1, 0.000_000_000_11, 0.000_000_000_09]),
//			0.000_000_000_008_165,
//			0.000_000_000_000_001
//		));
//}

