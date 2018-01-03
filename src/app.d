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

import std.stdio;
import std.random : Mt19937_64, unpredictableSeed;
import std.random    : uniform;
import std.range     : generate, take;
import std.csv;
import std.file;
import std.array;
import std.algorithm;
import std.typecons;
import std.parallelism;

import network;
import evolution;

void main()
{
	import std.random : Mt19937_64, unpredictableSeed;
	auto rng = Mt19937_64(unpredictableSeed());
	
	SpecimenParams sp;
	sp.inputs  = 2;
	sp.outputs = 1;
	sp.layers  = 2;
	sp.neurons = 2;
	sp.weights.min = -10;
	sp.weights.max =  10;
	
//	Evolution!Network evolution;
	
	auto file = File("tests.csv", "r");
	
//	auto data = file.byLine.joiner("\n").csvReader!(Tuple!(double, double, double, double)).array;
	
	double[][] InputData;
	double[][] OutputData;
	
	foreach (row; file.byLine.joiner("\n").csvReader!(Tuple!(double, double, double, double)))
	{
		InputData  ~= [ row[0], row[1] ];
		OutputData ~= [ row[2] ];
	}

//	foreach (i, val; taskPool.parallel(new int[population.organisms.length]))
//	{
//		for (ulong j = 0; j < InputData.length; j++)
//			population.organisms[i].fitness -= (OutputData[j][0] - population.organisms[i]( InputData[j] )[0]) * (OutputData[j][0] - population.organisms[i](InputData[j])[0]);
//		
//		writeln("network = ", population.organisms[i].fitness);
//	}
}
