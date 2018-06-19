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

// Standard D modules
import core.time              : Duration, minutes, seconds;
import std.algorithm          : count;
import std.conv               : to;
import std.csv                : csvReader;
import std.datetime.stopwatch : StopWatch;
import std.exception          : enforce;
import std.file               : readText;
import std.getopt             : defaultGetoptPrinter, getopt;
import std.math               : isFinite;
import std.random             : unpredictableSeed;
import std.range              : zip;
import std.stdio              : stdout, write, writeln;
import std.string             : format;

// Cuda modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common               : ANSIColor, ansiFormat, humanReadable;
import evolution.population : NetworkParams, Population;
import math.matrix          : Matrix, transpose;
import math.random          : RandomPool;
import math.statistics      : MAE, MASE, MPE;


void main(string[] args)
{
	uint   device;      /// Device to use.
	uint   minuteLimit; /// Time cap for ANN to train. For convenience is integer, minutes.
	string pathToData;  /// Path to the folder cointaining datasets. Must be of the specific structure.
	
	// Network parameters
	uint  layers;
	uint  neurons;
	float min;
	float max;
	
	
	immutable helpString = "Use " ~ args[0] ~ " --help for help.";
	
	auto opts = getopt(
		args,
		"path",      "Path to data directory", &pathToData,
		"device|d",  "A device to use.",       &device,
		"time|t",    "Time limit, minutes.",   &minuteLimit,
		"layers|l",  "Number of layers.",      &layers,
		"neurons|n", "Number of neurons.",     &neurons,
		"min",       "Minimum weight.",        &min,
		"max",       "Maximum weight.",        &max
	);
	
	if (opts.helpWanted)
	{
		defaultGetoptPrinter("\n\tThis is SteamSpyder. It collects data from Steam\n", opts.options);
		return;
	}
	
	enforce(device < cudaGetDeviceCount(), "%d GPU device is used, but only %d found.".format(device, cudaGetDeviceCount()));
	
	enforce(minuteLimit >= 1, "Time limit is set to %d minute(s), but must be at leats 1 minute.".format(minuteLimit));
	enforce(neurons     >= 2, "Neural network must have at leas 2 neurons, but got %d.".format(neurons));
	enforce(layers      >= 2, "Neural network must have at leas 2 layers, but got %d.".format(layers));
	
	enforce(isFinite(min), "Minimum weigth must be finite, but got %f".format(min));
	enforce(isFinite(max), "Maximum weigth must be finite, but got %f".format(max));
	
	enforce(max >= min, "Minimum weight %f is greater than maximum weight %f.".format(min, max));
	
	cudaSetDevice(device);
	
	immutable timeLimit            = minuteLimit.minutes();
	immutable populationMultiplier = 10;
	
	immutable seed = unpredictableSeed();
	writeln("Random seed = %d".ansiFormat(ANSIColor.white).format(seed));
	
	auto curandGenerator = CurandGenerator(curandRngType_t.PSEUDO_DEFAULT, seed);
	scope(exit) curandGenerator.destroy;
	
	auto pool = RandomPool(curandGenerator);
	scope(exit) pool.freeMem();
	
	cublasHandle_t cublasHandle;
	cublasCreate(cublasHandle);
	scope(exit) cublasDestroy(cublasHandle);
	
	auto trainingInputs = Matrix(readText(pathToData ~ "/training/inputs.csv"));
	scope(exit) trainingInputs.freeMem();
	
	auto trainingOutputs = Matrix(readText(pathToData ~ "/training/outputs.csv"));
	scope(exit) trainingOutputs.freeMem();
	
	auto validationInputs = Matrix(readText(pathToData ~ "/validation/inputs.csv"));
	scope(exit) validationInputs.freeMem();
	
	auto validationOutputs = Matrix(readText(pathToData ~ "/validation/outputs.csv"));
	scope(exit) validationOutputs.freeMem();
	
	NetworkParams params = {
		inputs  : trainingInputs.cols,
		neurons : neurons,
		outputs : trainingOutputs.cols,
		layers  : layers,
		min     : min,
		max     : max
	};
	
	writeln(
		("\tNetwork parameters: "
			~ "%d".ansiFormat(ANSIColor.white) ~ " inputs, "
			~ "%d".ansiFormat(ANSIColor.white) ~ " hidden neurons, "
			~ "%d".ansiFormat(ANSIColor.white) ~ " outputs, "
			~ "%d".ansiFormat(ANSIColor.white) ~ " layers, "
			~ "minimum weigth is " ~ "%f ".ansiFormat(ANSIColor.white)
			~ "maximun weight is " ~ "%f. ".ansiFormat(ANSIColor.white)
			~ "In total %d degrees of freedom"
		).format(
			params.inputs,
			params.neurons,
			params.outputs,
			params.layers,
			params.min,
			params.max,
			params.degreesOfFreedom
	));
	
	writeln(
		("\tPopulation multiplier = %d. Total population size is "
			~ "%d".ansiFormat(ANSIColor.white) ~ " networks"
		).format(populationMultiplier, populationMultiplier * params.degreesOfFreedom)
	);
	
	write("\tGenerating population...");
	stdout.flush();
	
	auto population = Population(params, params.degreesOfFreedom * populationMultiplier, pool);
	scope(exit) population.freeMem();
	
	writeln(" [ " ~ "done".ansiFormat(ANSIColor.green) ~ " ]");
	writeln("\tEstimated memory usage: " ~ population.size.humanReadable.ansiFormat(ANSIColor.white));
	
	writeln("\tTime limit is set to " ~ "%s".ansiFormat(ANSIColor.white).format(timeLimit));
	
	StopWatch stopWatch;
	stopWatch.start();
	while (true)
	{
		write("\tGeneration #%d: ".format(population.generation).ansiFormat(ANSIColor.white));
		stdout.flush();
		
		population.fitness(trainingInputs, trainingOutputs, cublasHandle);
		
		writeln(
			("\tbest = " ~ "%f".ansiFormat(ANSIColor.brightGreen)
				~ "\tworst = " ~ "%f".ansiFormat(ANSIColor.brightRed)
				~ "\tmean = " ~ "%f".ansiFormat(ANSIColor.brightYellow)
				~ "\t%s".ansiFormat(ANSIColor.white).format(timeLimit - stopWatch.peek()) ~ " left"
			).format(
				population.best.fitness,
				population.worst,
				population.mean
		));
		
		if (stopWatch.peek() >= timeLimit)
		{
			writeln("\tExceeded time limit [ " ~ "stopped".ansiFormat(ANSIColor.red) ~ " ]");
			break;
		}
		
		if (population.best.fitness <= 0.05)
		{
			writeln("\tAcceptable solution found [ " ~ "stopped".ansiFormat(ANSIColor.green) ~ " ]");
			break;
		}
		
		population.evolve(pool);
	}
	stopWatch.stop();
	stopWatch.reset();
	
	// TODO: Check validation data
	auto trainingOutputsT = Matrix(trainingOutputs.cols, trainingOutputs.rows);
	scope(exit) trainingOutputsT.freeMem();
	
	auto validationOutputsT = Matrix(validationOutputs.cols, validationOutputs.rows);
	scope(exit) validationOutputsT.freeMem();
							
	auto approx = Matrix(trainingOutputs.rows, trainingOutputs.cols);
	scope(exit) approx.freeMem();
	
	auto approxT = Matrix(trainingOutputs.cols, trainingOutputs.rows); // MASE operates on transposed matrices
	scope(exit) approxT.freeMem();
	
	transpose(trainingOutputs,   trainingOutputsT,   cublasHandle);
	transpose(validationOutputs, validationOutputsT, cublasHandle);
	
	population.best()(trainingInputs, approx, cublasHandle);
	transpose(approx, approxT, cublasHandle);
	
	float tMase = MASE(trainingOutputsT, approxT, cublasHandle);
	float tMae = MAE(trainingOutputsT, approxT, cublasHandle);
	float tMpe = MPE(trainingOutputsT, approxT, cublasHandle);
	
	population.best()(validationInputs, approx, cublasHandle);
	transpose(approx, approxT, cublasHandle);
	
	float vMase = MASE(validationOutputsT, approxT, cublasHandle);
	float vMae = MAE(validationOutputsT, approxT, cublasHandle);
	float vMpe = MPE(validationOutputsT, approxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
	foreach (j, v; approx)
		writeln("\t>>> Expected %f, got %f".format(trainingOutputs[j], approx[j]));
	
	writeln("Training results:".ansiFormat(ANSIColor.white));
	writeln("\tTraning dataset:"
		~ " best MASE = " ~ "%f".ansiFormat(ANSIColor.white).format(tMase)
		~ ", best MAE = " ~ "%f".ansiFormat(ANSIColor.white).format(tMae)
		~ ", best MPE = " ~ "%f%%".ansiFormat(ANSIColor.white).format(tMpe)
	);
	writeln("\tValidation dataset:"
		~ " best MASE = " ~ "%f".ansiFormat(ANSIColor.white).format(vMase)
		~ ", best MAE = " ~ "%f".ansiFormat(ANSIColor.white).format(vMae)
		~ ", best MPE = " ~ "%f%%".ansiFormat(ANSIColor.white).format(vMpe)
	);
	writeln("Best solution so far is ", population.best());
}

