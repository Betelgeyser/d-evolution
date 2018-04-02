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
module common;

import core.stdc.stdlib : malloc, free;


version (unittest)
{
	public import std.stdio  : write, writeln;
	import std.traits : fullyQualifiedName;
	
	template writetest(alias T)
	{
		enum writetest = "write(\"\x1b[1m" ~ fullyQualifiedName!T ~ "...\x1b[0m\");
			scope(failure) writeln(\"\x1b[1;31m failure!\x1b[0m\");
			scope(success) writeln(\"\x1b[1;32m success =)\x1b[0m\");";
	}
	
	template notTested(alias T)
	{
		enum notTested = "writeln(\"\x1b[1m" ~ fullyQualifiedName!T ~ "... \x1b[33mnot tested.\x1b[0m\");";
	}
}

T[] nogcMalloc(T)(ulong items) nothrow @nogc
{
	return (cast(T*)malloc(items * T.sizeof))[0 .. items];
}

void free(T)(ref T[] array)
{
	free(array.ptr);
}