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
import std.datetime.stopwatch : StopWatch;
import std.file               : readText;
import std.getopt             : defaultGetoptPrinter, getopt;
import std.math               : isFinite, lround;
import std.random             : unpredictableSeed;
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
	uint   device;         /// GPU device to use.
	uint   timeLimit;      /// Time limit to train ANN, seconds.
	uint   populationSize; /// Population size.
	uint   report;         /// Report every X generation.
	string pathToData;     /// Path to the folder cointaining datasets. Must be of the specific structure.
	
	// Network parameters
	uint  layers;
	uint  neurons;
	float min;
	float max;
	
	immutable helpString = "Use " ~ args[0] ~ " --help for help.";
	
	auto opts = getopt(
		args,
		"path",         "Path to data directory.",                   &pathToData,
		"device|d",     "GPU device to use.",                        &device,
		"time|t",       "Time limit, seconds.",                      &timeLimit,
		"layers|l",     "Number of layers.",                         &layers,
		"neurons|n",    "Number of neurons.",                        &neurons,
		"population|p", "Population multiplier.",                    &populationSize,
		"report|r",     "Print training report every X generations", &report,
		"min",          "Minimum connection weight.",                &min,
		"max",          "Maximum connection weight.",                &max
	);
	
	if (opts.helpWanted)
	{
		defaultGetoptPrinter("\n\tDNN is D Neural Network. It uses neuroevolution to learn.\n", opts.options);
		return;
	}
	
	if (device >= cudaGetDeviceCount())
	{
		writeln("%d GPU device is not found.".format(device));
		return;
	}
	
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
	
	auto population = Population(params, populationSize, pool);
	scope(exit) population.freeMem();
	
	writeln(" [ " ~ "done".ansiFormat(ANSIColor.green) ~ " ]");
	writeln("\tEstimated memory usage: " ~ population.size.humanReadable.ansiFormat(ANSIColor.white));
	
	writeln("\tTime limit is set to " ~ "%s".ansiFormat(ANSIColor.white).format(timeLimit.seconds()));
	
	StopWatch stopWatch;
	stopWatch.start();
	while (true)
	{
		/// Caching generation report prevents output from fast scrolling which causes your eyes bleed.
		string reportString = "\tGeneration #%d:".format(population.generation).ansiFormat(ANSIColor.white);
		
		// Not the smartest thing, but Ok for now.
		population.fitness(validationInputs, validationOutputs, cublasHandle);
		
		reportString ~= ("\n\t\tV: best = " ~ "%e".ansiFormat(ANSIColor.green)
			~ "\tmean = " ~ "%e".ansiFormat(ANSIColor.yellow)
			~ "\tworst = " ~ "%e".ansiFormat(ANSIColor.red)
		).format(
			population.best.fitness,
			population.mean,
			population.worst
		);
		
		population.fitness(trainingInputs, trainingOutputs, cublasHandle);
		
		reportString ~= ("\n\t\tT: best = ".ansiFormat(ANSIColor.white) ~ "%e".ansiFormat(ANSIColor.brightGreen)
			~ "\tmean = ".ansiFormat(ANSIColor.white) ~ "%e".ansiFormat(ANSIColor.brightYellow)
			~ "\tworst = ".ansiFormat(ANSIColor.white) ~ "%e".ansiFormat(ANSIColor.brightRed)
		).format(
			population.best.fitness,
			population.mean,
			population.worst
		);
		
		reportString ~= "\n\t\t%s".format(timeLimit.seconds() - stopWatch.peek()) ~ " left\n";
		
		if (report == 0 || population.generation % report == 0)
			writeln(reportString);
		
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
		
		import std.math : approxEqual;
		if (approxEqual(population.best.fitness, population.worst) && approxEqual(population.best.fitness, population.mean))
		{
			writeln("\tGenetic divercity lost [ " ~ "stopped".ansiFormat(ANSIColor.red) ~ " ]");
			break;
		}
		
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

