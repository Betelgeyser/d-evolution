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
module cuda.curand.exp;

import cuda.curand.types;

extern(C) package nothrow @nogc:
	curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type);
	curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
	curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, ulong seed);
	curandStatus_t curandGenerate(curandGenerator_t generator, float* outputPtr, size_t num);
	curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num);

