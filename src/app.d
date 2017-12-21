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

import std.stdio;
import std.random : Mt19937_64, unpredictableSeed;
import std.random    : uniform;
import std.range     : generate, take;
import std.array;

import network;

void main()
{
	auto rng = Mt19937_64(unpredictableSeed());
	
	Genome g;
	
	// Input layer
	g.input = 2;
	
	// First hidden layer
	double[][] tmp_1;
	for (long i = 0; i < 3; i++)
	{
		tmp_1 ~= generate(
			() => uniform!"[]"(-10.0, 10.0, rng)
		).take(2 + 1)
		 .array;
	}
	g.hidden ~= tmp_1;
	
	for (ulong i = 0; i < 2; i++)
	{
		double[][] tmp_2;
		for (ulong j = 0; j < 3; j++)
		{
			tmp_2 ~= generate(
				() => uniform!"[]"(-10.0, 10.0, rng)
			).take(3 + 1)
			 .array;
		}
		g.hidden ~= tmp_2;
	}
	
	// First hidden layer
	double[][] tmp_3;
	for (long i = 0; i < 1; i++)
	{
		tmp_3 ~= generate(
			() => uniform!"[]"(-10.0, 10.0, rng)
		).take(2 + 1)
		 .array;
	}
	g.output ~= tmp_3;
	
//	g.output ~= generate(
//		() => uniform!"[]"(-10.0, 10.0, rng)
//	).take(3 + 1)
//	 .array;
	
	writeln(g);
//	auto rn = Network(2, 1, 3, 3, -10, 10, rng);
//	writeln(rn);
//	
//	writeln(rn([4, 6]));
//	writeln("Hello World.");
}
