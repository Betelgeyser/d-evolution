/**
 * Copyright © 2018 Sergei Iurevich Filippov, All Rights Reserved.
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
module cuda.curand.functions;

import cuda.common;
import cuda.curand.types;
static import curand = cuda.curand.exp;

import core.stdc.stdlib  : malloc, free;
import std.exception     : enforce;


/**
 * Higer level wrapper around cuRAND generator. It provides D-style access to functions on a curandGenerator_t.
 *
 * TODO: NOT THREAD SAFE!!!
 */
struct CurandGenerator
{
	private
	{
		curandGenerator_t _generator;
	}
	
	this(in curandRngType_t rng_type, in ulong seed = 0) nothrow
	{
		scope(failure) freeMem();
		
		enforceCurand(curand.curandCreateGenerator(&_generator, rng_type));
		
		setPseudoRandomGeneratorSeed(seed);
	}
	
	void freeMem() nothrow
	{
		enforceCurand(curand.curandDestroyGenerator(_generator));
	}
	
	void setPseudoRandomGeneratorSeed(in ulong seed) nothrow
	{
		enforceCurand(curand.curandSetPseudoRandomGeneratorSeed(_generator, seed));
	}
	
	void generate(uint* outputPtr, in size_t num) nothrow
	{
		enforceCurand(curand.curandGenerate(_generator, outputPtr, num));
	}

	void generateUniform(float* outputPtr, in size_t num) nothrow
	{
		enforceCurand(curand.curandGenerateUniform(_generator, outputPtr, num));
	}
}

package void enforceCurand(curandStatus_t status) nothrow pure @safe
{
	if (status != curandStatus_t.SUCCESS)
		throw new Error(status.toString);
}

