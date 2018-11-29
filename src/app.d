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
import std.math               : isFinite, lround;
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
	version (unittest) {} else {
	uint   device;               /// GPU device to use.
	uint   timeLimit;            /// Time limit to train ANN, seconds.
	float  populationMultiplier; /// Population multiplier.
	string pathToData;           /// Path to the folder cointaining datasets. Must be of the specific structure.
	
	// Network parameters
	uint  layers;
	uint  neurons;
	float min;
	float max;
	
	immutable helpString = "Use " ~ args[0] ~ " --help for help.";
	
	auto opts = getopt(
		args,
		"path",          "Path to data directory.",    &pathToData,
		"device|d",      "GPU device to use.",         &device,
		"time|t",        "Time limit, seconds.",       &timeLimit,
		"layers|l",      "Number of layers.",          &layers,
		"neurons|n",     "Number of neurons.",         &neurons,
		"multiplier|m",  "Population multiplier.",     &populationMultiplier,
		"min",           "Minimum connection weight.", &min,
		"max",           "Maximum connection weight.", &max
	);
	
	if (opts.helpWanted)
	{
		defaultGetoptPrinter("\n\tThis is SteamSpyder. It collects data from Steam\n", opts.options);
		return;
	}
	
	enforce(device < cudaGetDeviceCount(), "%d GPU device is used, but only %d found.".format(device, cudaGetDeviceCount()));
	
	enforce(timeLimit >= 1, "Time limit is set to %d minute(s), but must be at leats 1 second.".format(timeLimit));
	enforce(neurons   >= 2, "Neural network must have at leas 2 neurons, but got %d.".format(neurons));
	enforce(layers    >= 2, "Neural network must have at leas 2 layers, but got %d.".format(layers));
	
	enforce(isFinite(min), "Minimum weigth must be finite, but got %f".format(min));
	enforce(isFinite(max), "Maximum weigth must be finite, but got %f".format(max));
	
	enforce(max >= min, "Minimum weight %f is greater than maximum weight %f.".format(min, max));
	
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
		("\tPopulation multiplier = %g. Total population size is "
			~ "%d".ansiFormat(ANSIColor.white) ~ " networks"
		).format(populationMultiplier, lround(populationMultiplier * params.degreesOfFreedom))
	);
	
	write("\tGenerating population...");
	stdout.flush();
	
	auto population = Population(params, lround(params.degreesOfFreedom * populationMultiplier), pool);
	scope(exit) population.freeMem();
	
	writeln(" [ " ~ "done".ansiFormat(ANSIColor.green) ~ " ]");
	writeln("\tEstimated memory usage: " ~ population.size.humanReadable.ansiFormat(ANSIColor.white));
	
	writeln("\tTime limit is set to " ~ "%s".ansiFormat(ANSIColor.white).format(timeLimit.seconds()));
	
	StopWatch stopWatch;
	stopWatch.start();
	while (true)
	{
		write("\tGeneration #%d:".format(population.generation).ansiFormat(ANSIColor.white));
		stdout.flush();
		
		population.fitness(trainingInputs, trainingOutputs, cublasHandle);
		
		writeln(
			("\tbest = " ~ "%e".ansiFormat(ANSIColor.brightGreen)
				~ "\tworst = " ~ "%e".ansiFormat(ANSIColor.brightRed)
				~ "\tmean = " ~ "%e".ansiFormat(ANSIColor.brightYellow)
				~ "\t%s".ansiFormat(ANSIColor.white).format(timeLimit.seconds() - stopWatch.peek()) ~ " left"
			).format(
				population.best.fitness,
				population.worst,
				population.mean
		));
		
		if (stopWatch.peek() >= timeLimit.seconds())
		{
			writeln("\tExceeded time limit [ " ~ "stopped".ansiFormat(ANSIColor.red) ~ " ]");
			break;
		}
		
		if (population.best.fitness <= 0.1)
		{
			writeln("\tAcceptable solution found [ " ~ "stopped".ansiFormat(ANSIColor.green) ~ " ]");
			break;
		}
		
//		import std.math : approxEqual;
//		if (approxEqual(population.best.fitness, population.worst) && approxEqual(population.best.fitness, population.mean))
//		{
//			writeln("\tAcceptable solution found [ " ~ "stopped".ansiFormat(ANSIColor.green) ~ " ]");
//			break;
//		}
		
		population.evolve(pool);
	}
	stopWatch.stop();
	stopWatch.reset();
	
	auto trainingOutputsT = Matrix(trainingOutputs.cols, trainingOutputs.rows);
	scope(exit) trainingOutputsT.freeMem();
	
	auto validationOutputsT = Matrix(validationOutputs.cols, validationOutputs.rows);
	scope(exit) validationOutputsT.freeMem();
	
	auto trainingApprox = Matrix(trainingOutputs.rows, trainingOutputs.cols);
	scope(exit) trainingApprox.freeMem();
	
	auto trainingApproxT = Matrix(trainingOutputs.cols, trainingOutputs.rows);
	scope(exit) trainingApproxT.freeMem();
	
	auto validationApprox = Matrix(validationOutputs.rows, validationOutputs.cols);
	scope(exit) validationApprox.freeMem();
	
	auto validationApproxT = Matrix(validationOutputs.cols, validationOutputs.rows);
	scope(exit) validationApproxT.freeMem();
	
	population.best()(trainingInputs, trainingApprox, cublasHandle);
	
	transpose(trainingOutputs, trainingOutputsT, cublasHandle);
	transpose(trainingApprox, trainingApproxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
	float tMase = MASE(trainingOutputsT, trainingApproxT, cublasHandle);
	float tMae  = MAE(trainingOutputsT, trainingApproxT, cublasHandle);
	float tMpe  = MPE(trainingOutputsT, trainingApproxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
	population.best()(validationInputs, validationApprox, cublasHandle);
	transpose(validationApprox, validationApproxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
	float vMase = MASE(validationOutputsT, validationApproxT, cublasHandle);
	float vMae  = MAE(validationOutputsT, validationApproxT, cublasHandle);
	float vMpe  = MPE(validationOutputsT, validationApproxT, cublasHandle);
	
	cudaDeviceSynchronize();
	
//	auto l2Outputs = Matrix(1, trainingOutputs.rows);
//	scope(exit) l2Outputs.freeMem();
//	
//	auto l2Approx = Matrix(1, approx.rows);
//	scope(exit) l2Approx.freeMem();
//	
//	cudaL2(trainingOutputsT, l2Outputs.values);
//	cudaL2(approxT, l2Approx.values);
	
	cudaDeviceSynchronize();
//	
//	writeln();
//	writeln("Ot,Ht,Lt,Ct,L2t,Oa,Ha,La,Ca,L2a,AE,PE");
//	for (int i = 0; i < approx.rows; ++i)
//	{
//		writeln("%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g".format(
//			trainingOutputs[i, 0],
//			trainingOutputs[i, 1],
//			trainingOutputs[i, 2],
//			trainingOutputs[i, 3],
//			l2Outputs[0, i],
//			approx[i, 0],
//			approx[i, 1],
//			approx[i, 2],
//			approx[i, 3],
//			l2Approx[0, i],
//			abs(l2Outputs[0, i] - l2Approx[0, i]),
//			abs(l2Outputs[0, i] - l2Approx[0, i]) / l2Outputs[0, i] * 100.0
//		));
//	}
	
	writeln();
	writeln("Training results:".ansiFormat(ANSIColor.white));
	writeln("\tTraning dataset:"
		~ " best MASE = " ~ "%g".ansiFormat(ANSIColor.white).format(tMase)
		~ ", best MAE = " ~ "%g".ansiFormat(ANSIColor.white).format(tMae)
		~ ", best MPE = " ~ "%g%%".ansiFormat(ANSIColor.white).format(tMpe)
	);
	writeln("\tValidation dataset:"
		~ " best MASE = " ~ "%g".ansiFormat(ANSIColor.white).format(vMase)
		~ ", best MAE = " ~ "%g".ansiFormat(ANSIColor.white).format(vMae)
		~ ", best MPE = " ~ "%g%%".ansiFormat(ANSIColor.white).format(vMpe)
	);
	writeln("Best solution so far is ", population.best());
}}

