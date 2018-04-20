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
 *
 * This package provides access to CUDA cuRAND.
 */
module cuda.curand;

import cuda.common;
import cuda.curand.exp;

public import cuda.curand.types;

/**
 * Higer level wrapper around cuRAND generator. It provides D-style access to functions on a curandGenerator_t.
 */
struct CurandGenerator
{
	private curandGenerator_t _generator;
	
	this(curandRngType_t rng_type, ulong seed = 0) nothrow @nogc
	{
		enforceCurand(curandCreateGenerator(&_generator, rng_type));
		setPseudoRandomGeneratorSeed(seed);
	}
	
	void destroy() nothrow @nogc
	{
		enforceCurand(curandDestroyGenerator(_generator));
	}
	
	void setPseudoRandomGeneratorSeed(ulong seed) nothrow @nogc
	{
		enforceCurand(curandSetPseudoRandomGeneratorSeed(_generator, seed));
	}
	
	void generate(float* outputPtr, size_t num) nothrow @nogc
	{
		enforceCurand(curandGenerate(_generator, outputPtr, num));
	}

	void generateUniform(float* outputPtr, size_t num) nothrow @nogc
	{
		enforceCurand(curandGenerateUniform(_generator, outputPtr, num));
	}
}

package void enforceCurand(curandStatus_t error) pure nothrow @safe @nogc
{
	assert (error == curandStatus_t.SUCCESS, error.toString);
}

