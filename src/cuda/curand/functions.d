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

import cuda.curand.types;

extern(C):
	curandStatus_t curandCreateGenerator(ref curandGenerator_t generator, curandRngType_t rng_type) nothrow @nogc;
	curandStatus_t curandCreateGeneratorHost(ref curandGenerator_t generator, curandRngType_t rng_type) nothrow @nogc;
	curandStatus_t curandDestroyGenerator(curandGenerator_t generator) nothrow @nogc;
	
	curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, ulong seed) nothrow @nogc;
	
	curandStatus_t curandGenerate(curandGenerator_t generator, float* outputPtr, size_t num) nothrow @nogc;
	curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num) nothrow @nogc;

	curandStatus_t curandGetVersion(int* v) nothrow @nogc;