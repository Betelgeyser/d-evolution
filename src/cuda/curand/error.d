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
module cuda.curand.error;

import std.conv      : to;
import std.exception : Error;

import cuda.curand.types;

class CurandError : Error
{
	this(curandStatus_t curandStatus, string file = __FILE__, size_t line = __LINE__)
	{
		super(curandStatus.to!string, file, line);
	}
}

/**
 * D-style cuRAND error check.
 */
pragma(inline, true) void enforceCurand(curandStatus_t curandStatus, string file = __FILE__, size_t line = __LINE__)
{
	if (curandStatus != curandStatus_t.CURAND_STATUS_SUCCESS)
		throw new CurandError(curandStatus, file, line);
}

