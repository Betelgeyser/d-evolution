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
module cuda.curand.types;

alias curandGenerator_t = curandGenerator*;
package struct curandGenerator;

enum curandStatus_t
{
	CURAND_STATUS_SUCCESS                   = 0,
	CURAND_STATUS_VERSION_MISMATCH          = 100,
	CURAND_STATUS_NOT_INITIALIZED           = 101,
	CURAND_STATUS_ALLOCATION_FAILED         = 102,
	CURAND_STATUS_TYPE_ERROR                = 103,
	CURAND_STATUS_OUT_OF_RANGE              = 104,
	CURAND_STATUS_LENGTH_NOT_MULTIPLE       = 105,
	CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
	CURAND_STATUS_LAUNCH_FAILURE            = 201,
	CURAND_STATUS_PREEXISTING_FAILURE       = 202,
	CURAND_STATUS_INITIALIZATION_FAILED     = 203,
	CURAND_STATUS_ARCH_MISMATCH             = 204,
	CURAND_STATUS_INTERNAL_ERROR            = 999
}

enum curandRngType_t
{
	CURAND_RNG_TEST                    = 0,
	CURAND_RNG_PSEUDO_DEFAULT          = 100,
	CURAND_RNG_PSEUDO_XORWOW           = 101,
	CURAND_RNG_PSEUDO_MRG32K3A         = 121,
	CURAND_RNG_PSEUDO_MTGP32           = 141,
	CURAND_RNG_PSEUDO_MT19937          = 142,
	CURAND_RNG_PSEUDO_PHILOX4_32_10    = 161,
	CURAND_RNG_QUASI_DEFAULT           = 200,
	CURAND_RNG_QUASI_SOBOL32           = 201,
	CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
	CURAND_RNG_QUASI_SOBOL64           = 203,
	CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
}

