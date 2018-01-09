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
import std.datetime;

import dnn.nn.network;
import dnn.ga.genome;
import dnn.ga.population;

void main()
{
	SpecimenParams sp;
	sp.inputs  = 1;
	sp.outputs = 1;
	sp.layers  = 3;
	sp.neurons = 3;
	sp.weights.min = -1_000_000_000;
	sp.weights.max =  1_000_000_000;
	
	Population!Network population;
	
	auto file = File("tests.csv", "r");
	
	double[][] inputData;
	double[][] outputData;
	
	foreach (row; file.byLine.joiner("\n").csvReader!(Tuple!(double, double, double, double, double)))
	{
		inputData  ~= [ row[0] ];
		outputData ~= [ row[4] ];
	}
	
	population.loadData(inputData, outputData);
	population.specimenParams = sp;
	
	StopWatch sw;
	sw.start();
	population.populate(1000);
	writeln("Initial population:");
	writeln(">>> Best fitness = ", population.bestFitness, "; avg fitness = ", population.avgFitness);
	
	writeln(sw.peek().msecs(), " msec");
	
	ulong counter;
	do
	{
		counter++;
		population.selection();
		writeln("Generation ", counter, ":");
		writeln(">>> Best = ", population.bestFitness, "; worst = ", population.worstFitness, "; avg = ", population.avgFitness);
	
		writeln(sw.peek().msecs(), " msec");
	} while (population.bestFitness > 0.01 && counter < 10_000);
	sw.stop();
}
