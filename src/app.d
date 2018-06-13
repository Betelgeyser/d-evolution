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
import std.file               : readText;
import std.range              : zip;
import std.stdio              : stdout, write, writeln;
import std.string             : format;
import std.typecons           : tuple;

// Cuda modules
import cuda.cudaruntimeapi;
import cuda.cublas;
import cuda.curand;

// DNN modules
import common               : ANSIColor, ansiFormat, humanReadable;
import evolution.population : NetworkParams, Population;
import math.matrix          : Matrix, transpose;
import math.random          : RandomPool;
import math.statistics      : MAE, MASE;


void main()
{
	StopWatch stopWatch;
	
	uint[]  inputs  = [30, 7, 1];//, 90];
	uint[]  outputs = [1];//, 7];//, 30];
	uint[]  layers  = [2];//,  3];//,  4,  5,  6,  7, 8, 9, 10, 11, 12];
	uint[]  neurons = [2, 4, 6, 8, 10];//, 15, 20];
	float[] min     = [-1.0e3, -1.0e6, -1.0e9];
	float[] max     = [ 1.0e3,  1.0e6,  1.0e9];
	
	uint[] populationMultiplier = [10];//[1, 5, 10];//, 15, 20];
	
	auto curandGenerator = CurandGenerator(curandRngType_t.PSEUDO_DEFAULT);
	scope(exit) curandGenerator.destroy;
	
	auto pool = RandomPool(curandGenerator);
	scope(exit) pool.freeMem();
	
	cublasHandle_t cublasHandle;
	cublasCreate(cublasHandle);
	scope(exit) cublasDestroy(cublasHandle);
	
	ulong trialNumber = 0;
	
	auto timeLimit       = 15.minutes();
//	auto timeLimit       = 5.seconds();
	uint generationLimit = 5000;
	
	foreach (i; inputs)
		foreach (o; outputs)
			foreach (l; layers)
				foreach (n; neurons)
					foreach (mul; populationMultiplier)
						foreach (boundaries; zip(min, max))
						{
							writeln("Setting up trial #%d".format(++trialNumber).ansiFormat(ANSIColor.white));
							
							NetworkParams params = {
								inputs  : i * 7,
								neurons : n,
								outputs : o * 4,
								layers  : l,
								min     : boundaries[0],
								max     : boundaries[1]
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
								).format(mul, mul * params.degreesOfFreedom)
							);
							
							write("\tGenerating population...");
							stdout.flush();
							
							auto population = Population(params, params.degreesOfFreedom * mul, pool);
							scope(exit) population.freeMem();
							
							writeln(" [ " ~ "done".ansiFormat(ANSIColor.green) ~ " ]");
							writeln("\tEstimated memory usage: " ~ population.size.humanReadable.ansiFormat(ANSIColor.white));
							
							auto trainingInputs = Matrix(readText("data/" ~ i.to!string ~ "/training/inputs.csv"));
							scope(exit) trainingInputs.freeMem();
							
							auto trainingOutputs = Matrix(readText("data/" ~ i.to!string ~ "/training/outputs.csv"));
							scope(exit) trainingOutputs.freeMem();
							
							auto testingInputs = Matrix(readText("data/" ~ i.to!string ~ "/testing/inputs.csv"));
							scope(exit) testingInputs.freeMem();
							
							auto testingOutputs = Matrix(readText("data/" ~ i.to!string ~ "/testing/outputs.csv"));
							scope(exit) testingOutputs.freeMem();
							
							writeln(
								("\tGeneration is capped at " ~ "%d".ansiFormat(ANSIColor.white)
									~ ", time limit is set to " ~ "%s".ansiFormat(ANSIColor.white)
								).format(generationLimit, timeLimit)
							);
							
							stopWatch.start();
							while (true)
							{
								write("\tGeneration #%d: ".format(population.generation).ansiFormat(ANSIColor.white));
								stdout.flush();
								
								population.fitness(trainingInputs, trainingOutputs, cublasHandle);
								
								writeln(
									("\tbest = " ~ "%f".ansiFormat(ANSIColor.brightGreen) ~
										"\tworst = " ~ "%f".ansiFormat(ANSIColor.brightRed) ~
										"\tmean = " ~ "%f".ansiFormat(ANSIColor.brightYellow)
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
								
								if (population.generation >= generationLimit)
								{
									writeln("\tGenerations cap reached [ " ~ "stopped".ansiFormat(ANSIColor.red) ~ " ]");
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
							
							auto approx = Matrix(trainingOutputs.rows, trainingOutputs.cols);
							scope(exit) approx.freeMem();
							
							auto approxT = Matrix(trainingOutputs.cols, trainingOutputs.rows); // MASE operates on transposed matrices
							scope(exit) approxT.freeMem();
							
							transpose(trainingOutputs, trainingOutputsT, cublasHandle);
							
							population.best()(trainingInputs, approx, cublasHandle);
							
							transpose(approx, approxT, cublasHandle);
							
							float mase = MASE(trainingOutputsT, approxT, cublasHandle);
							float mae = MAE(trainingOutputsT, approxT, cublasHandle);
							
							cudaDeviceSynchronize();
							
							foreach (j, v; approx)
								writeln("\t>>> Expected %f, got %f".format(trainingOutputs[j], approx[j]));
							
							writeln(
								("\tTrial results: ".ansiFormat(ANSIColor.white) ~
									"best MASE = " ~ "%f".ansiFormat(ANSIColor.brightGreen) ~
									", best MAE = " ~ "%f".ansiFormat(ANSIColor.brightGreen)
								).format(mase, mae)
							);
							writeln("Best solution so far is ", population.best());
						}
}

