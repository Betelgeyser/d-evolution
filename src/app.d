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
	sp.inputs  = 1;
	sp.outputs = 1;
	sp.layers  = 3;
	sp.neurons = 3;
	sp.weights.min = -100;
	sp.weights.max =  100;
	
	Population!Network population;
	
	auto file = File("tests.csv", "r");
	
	double[][] inputData;
	double[][] outputData;
	
	foreach (row; file.byLine.joiner("\n").csvReader!(Tuple!(double, double)))
	{
		inputData  ~= [ row[0] ];
		outputData ~= [ row[1] ];
	}
	
	population.loadData(inputData, outputData);
	population.specimenParams = sp;
	
	population.populate(1000, rng);
	writeln("Initial population:");
//	writeln(">>> population.fitness.values = ", population.fitness.values);
	writeln(">>> Best fitness = ", population.bestFitness, "; avg fitness = ", population.avgFitness);
	
	ulong counter;
	do
	{
		counter++;
		population.selection(rng);
		writeln("Generation ", counter, ":");
//		writeln(">>> population.fitness.values = ", population.fitness.values);
		writeln(">>> Best = ", population.bestFitness, "; worst = ", population.worstFitness, "; avg = ", population.avgFitness);
	} while (population.bestFitness > 0.01);
}
