/**
 * Copyright © 2017 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
import std.conv               : ConvException, to;
import std.datetime.stopwatch : StopWatch;
import std.file               : readText;
import std.getopt             : defaultGetoptPrinter, getopt;
import std.math               : approxEqual, isFinite, lround;
import std.random             : unpredictableSeed;
import std.stdio              : stderr, stdout, write, writeln;
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
import math.statistics      : MAE, MPE;

class CLIException : Exception
{
	this(string msg, string file = __FILE__, size_t line = __LINE__, Throwable nextInChain = null) @nogc nothrow pure @safe
	{
		super(msg, file, line, nextInChain);
	}
	
	this(string msg, Throwable nextInChain, string file = __FILE__, size_t line = __LINE__) @nogc nothrow pure @safe
	{
		super(msg, file, line, nextInChain);
	}
}

uint seed;   /// Seed for the PRNG.
uint device; /// GPU device to use.

void parseSeed(string option, string value) @safe
{
	if (value != "random") // "random" is a valid value
		try
			seed = value.to!uint;
		catch (ConvException e)
			throw new CLIException("seed must be a positive integer value or \"random\".", e);
}

void parseDevice(string option, string value)
{
	try
		device = value.to!uint;
	catch (ConvException e)
		throw new CLIException("invalid device id.");
	
	if (device >= cudaGetDeviceCount())
		throw new CLIException("invalid device id.");
}

void main(string[] args)
{
	version (unittest) {} else {
	seed = unpredictableSeed(); // Default runtime initialization. unpredictableSeed cannot be determined at compile time.
	uint   populationSize; /// Population size.
	uint   report;         /// Report every X generation.
	uint   timeLimit;      /// Time limit to train ANN, seconds.
	string fitnessString;  /// Error function.
	string pathToData;     /// Path to the folder cointaining datasets. Must be of the specific structure.
	
	// Network parameters
	uint  layers;
	uint  neurons;
	float min;
	float max;
	
	float function(in Matrix, in Matrix, cublasHandle_t) fitnessFunction = &MAE;
	
	immutable helpString = "Use " ~ args[0] ~ " --help for help.";
	
	try
	{
		auto opts = getopt(
			args,
			"path",         "Path to the data directory.",               &pathToData,
			"device|d",     "GPU device to use.",                        &parseDevice,
			"time|t",       "Time limit, seconds.",                      &timeLimit,
			"layers|l",     "Number of layers.",                         &layers,
			"neurons|n",    "Number of neurons.",                        &neurons,
			"population|p", "Population size.",                          &populationSize,
			"report|r",     "Print training report every X generations", &report,
			"fitness-function|f", "Fitness function. Available values: MAE (default), MPE", &fitnessString,
			"min",          "Minimum connection weight.",                &min,
			"max",          "Maximum connection weight.",                &max,
			"seed|s",       "Seed for the PRNG. May be set to a specific unsigned integer number or \"random\" (default).", &parseSeed
		);
		
		if (opts.helpWanted)
		{
			defaultGetoptPrinter("\n\tDNN is D Neural Network. It uses neuroevolution to learn.\n", opts.options);
			return;
		}
	}
	catch(CLIException e)
	{
		stderr.writeln("Abort: %s Use --help for more information.".format(e.msg));
		return;
	}
	
	writeln("PRNG seed is set to %d.".ansiFormat(ANSIColor.white).format(seed));
	
	if (neurons < 2) // Why 2? Isn't 1 sufficient?
	{
		writeln("Neural network must have at leas 2 neurons.");
		return;
	}
	
	if (layers < 2) // Why?
	{
		writeln("Neural network must have at leas 2 layers.");
		return;
	}
	
	if (!isFinite(min))
	{
		writeln("Minimum weigth must be a finite number, not %g".format(min));
		return;
	}
	
	if (!isFinite(max))
	{
		writeln("Maximum weigth must be a finite number, not %g".format(max));
		return;
	}
	
	if (min >= max)
	{
		writeln("The minimum weight %g must be less than the maximum weight %g.".format(min, max));
		return;
	}
	
	if (populationSize < 2)
	{
		writeln("Population must be at leats 2 individuals.");
		return;
	}
	
	if (fitnessString != "")
	{
		if (fitnessString == "MAE")
			fitnessFunction = &MAE;
		
		if (fitnessString == "MPE")
			fitnessFunction = &MPE;
		
//		if (fitnessString == "MASE")
//			fitnessFunction = &MASE;
	}
	
	cudaSetDevice(device);
	
	immutable seed = unpredictableSeed();
	writeln("Random seed = %d".ansiFormat(ANSIColor.white).format(seed));
	
	auto pool = RandomPool(curandRngType_t.PSEUDO_DEFAULT, seed);
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
			~ "minimum weigth is " ~ "%g ".ansiFormat(ANSIColor.white)
			~ "maximun weight is " ~ "%g. ".ansiFormat(ANSIColor.white)
		).format(
			params.inputs,
			params.neurons,
			params.outputs,
			params.layers,
			params.min,
			params.max
	));
	
	writeln(
		("\tPopulation size is " ~ "%d".ansiFormat(ANSIColor.white) ~ " networks")
			.format(populationSize)
	);
	
	write("\tGenerating population...");
	stdout.flush();
	
	auto population = Population(params, populationSize, pool, fitnessFunction);
	scope(exit) population.freeMem();
	
	writeln(" [ " ~ "done".ansiFormat(ANSIColor.green) ~ " ]");
	
	writeln("\tTime limit is set to " ~ "%s".ansiFormat(ANSIColor.white).format(timeLimit.seconds()));
	
	StopWatch stopWatch;
	stopWatch.start();
	while (true)
	{
		auto time = stopWatch.peek();
		string reportString; /// Caching generation report prevents output from fast scrolling which causes your eyes bleed.
		
		// Calculation error on training dataset. Evolution desision is based on it.
		population.fitness(trainingInputs, trainingOutputs, cublasHandle);
		
		if (report == 0 || population.generation % report == 0 || time >= timeLimit.seconds())
		{
			reportString = "\tGeneration #%d:".format(population.generation).ansiFormat(ANSIColor.white);
			
			reportString ~= ("\n\t\tT: best = ".ansiFormat(ANSIColor.white) ~ "%e".ansiFormat(ANSIColor.brightGreen)
				~ "    mean = ".ansiFormat(ANSIColor.white) ~ "%e".ansiFormat(ANSIColor.brightYellow)
				~ "    worst = ".ansiFormat(ANSIColor.white) ~ "%e".ansiFormat(ANSIColor.brightRed)
			).format(
				population.best.fitness,
				population.mean,
				population.worst
			);
			
			auto validationApprox = Matrix(validationOutputs.rows, validationOutputs.cols);
			scope(exit) validationApprox.freeMem();
			
			population.best()(validationInputs, validationApprox, cublasHandle);
			
			auto Error = fitnessFunction(validationOutputs, validationApprox, cublasHandle);
			
			cudaDeviceSynchronize();
			
			reportString ~= "\n\t\tV: best = " ~ "%e".ansiFormat(ANSIColor.green).format(Error);
			reportString ~= "\n\t\t%s".format(timeLimit.seconds() - time) ~ " left\n";
			
			writeln(reportString);
		}
		
		if (time >= timeLimit.seconds())
		{
			writeln("\tExceeded time limit [ " ~ "stopped".ansiFormat(ANSIColor.red) ~ " ]");
			break;
		}
		
		if (population.best.fitness <= 0.1)
		{
			writeln("\tAcceptable solution found [ " ~ "stopped".ansiFormat(ANSIColor.green) ~ " ]");
			break;
		}
		
//		if (approxEqual(population.best.fitness, population.worst, 1000) && approxEqual(population.best.fitness, population.mean, 1000))
//		{
//			writeln("\tGenetic divercity lost [ " ~ "stopped".ansiFormat(ANSIColor.red) ~ " ]");
//			break;
//		}
		
		population.evolve(pool);
	}
	stopWatch.stop();
	
//	auto trainingOutputsT = Matrix(trainingOutputs.cols, trainingOutputs.rows);
//	scope(exit) trainingOutputsT.freeMem();
//	
//	auto validationOutputsT = Matrix(validationOutputs.cols, validationOutputs.rows);
//	scope(exit) validationOutputsT.freeMem();
//	
	auto trainingApprox = Matrix(trainingOutputs.rows, trainingOutputs.cols);
	scope(exit) trainingApprox.freeMem();
//	
//	auto trainingApproxT = Matrix(trainingOutputs.cols, trainingOutputs.rows);
//	scope(exit) trainingApproxT.freeMem();
//	
	auto validationApprox = Matrix(validationOutputs.rows, validationOutputs.cols);
	scope(exit) validationApprox.freeMem();
//	
//	auto validationApproxT = Matrix(validationOutputs.cols, validationOutputs.rows);
//	scope(exit) validationApproxT.freeMem();
//	
	population.best()(trainingInputs, trainingApprox, cublasHandle);
	
//	transpose(trainingOutputs, trainingOutputsT, cublasHandle);
//	transpose(trainingApprox, trainingApproxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
//	float tMase = MASE(trainingOutputsT, trainingApproxT, cublasHandle);
//	float tMae  = MAE(trainingOutputsT, trainingApproxT, cublasHandle);
//	float tMpe  = MPE(trainingOutputsT, trainingApproxT, cublasHandle);
//	
//	cudaDeviceSynchronize();
//	
	population.best()(validationInputs, validationApprox, cublasHandle);
//	transpose(validationApprox, validationApproxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
	import std.json;
	auto result = population.best().json();
	writeln();
	writeln("Best solution so far is ", result.toJSON());
}}

