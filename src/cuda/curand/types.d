/**
 * Copyright © 2018 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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

package alias curandGenerator_t = curandGenerator*;
private struct curandGenerator;

enum curandStatus_t
{
	SUCCESS                   =   0,
	VERSION_MISMATCH          = 100,
	NOT_INITIALIZED           = 101,
	ALLOCATION_FAILED         = 102,
	TYPE_ERROR                = 103,
	OUT_OF_RANGE              = 104,
	LENGTH_NOT_MULTIPLE       = 105,
	DOUBLE_PRECISION_REQUIRED = 106,
	LAUNCH_FAILURE            = 201,
	PREEXISTING_FAILURE       = 202,
	INITIALIZATION_FAILED     = 203,
	ARCH_MISMATCH             = 204,
	INTERNAL_ERROR            = 999
}

enum curandRngType_t
{
	TEST                    =   0,
	PSEUDO_DEFAULT          = 100,
	PSEUDO_XORWOW           = 101,
	PSEUDO_MRG32K3A         = 121,
	PSEUDO_MTGP32           = 141,
	PSEUDO_MT19937          = 142,
	PSEUDO_PHILOX4_32_10    = 161,
	QUASI_DEFAULT           = 200,
	QUASI_SOBOL32           = 201,
	QUASI_SCRAMBLED_SOBOL32 = 202,
	QUASI_SOBOL64           = 203,
	QUASI_SCRAMBLED_SOBOL64 = 204
}

