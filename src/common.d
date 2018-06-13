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

// Standard D modules
import core.stdc.stdlib : malloc, free;
import std.math         : pow;
import std.string       : format;


/**
 * Manual GC-free memory allocation for D dynamic arrays.
 *
 * Bug-prone, use it carefully...
 */
T[] nogcMalloc(T)(ulong items) nothrow @nogc
{
	return (cast(T*)malloc(items * T.sizeof))[0 .. items];
}

/**
 * Manual GC-free memory releasing for D dynamic arrays.
 *
 * Bug-prone, use it carefully...
 */
void nogcFree(T)(ref T[] array) nothrow @nogc
{
	free(array.ptr);
	array.destroy();
}

/**
 * Returns: true if <math><mi>a</mi><mo>&le;</mo><mi>x</mi><mo>&le;</mo><mi>b</mi></math>.
 */
bool between(string boundaries = "[]")(in float x, in float a, in float b) @nogc nothrow pure @safe
in
{
	assert (a <= b, "common.between!\"" ~ boundaries ~ "\"(x, a, b): invalid boundary interval");
}
body
{
	static if (boundaries[0] == '[')
		enum opLeft = ">=";
	else static if (boundaries[0] == '(')
		enum opLeft = ">";
	else
		static assert (0, "common.between!\"" ~ boundaries ~ "\"(x, a, b): invalid boundary");
	
	static if (boundaries[1] == ']')
		enum opRight = "<=";
	else static if (boundaries[1] == ')')
		enum opRight = "<";
	else
		static assert (0, "common.between!\"" ~ boundaries ~ "\"(x, a, b): invalid boundary");
	
	return mixin("x" ~ opLeft ~ "a") && mixin("x" ~ opRight ~ "b");
}

/**
 * Returns: Arithmetic series
 *     <math><msub><mi>a</mi><mi>1</mi></msub><mo>+</mo><mi>...</mi><mo>+</mo><msub><mi>a</mi><mi>n</mi></msub></math>.
 *
 * Params:
 *     a1 = The first term of the sequence.
 *     an = The nth term.
 *     n = Number of terms in the sequence.
 */
float AS(in float a1, in float an, in float n) @nogc nothrow pure @safe
{
	return (a1 + an) * n / 2.0f;
}

string humanReadable(in float value) @safe
{
	immutable string spec = "%.1f";
	
	if (value < 1024.0)
		return (spec ~ " B").format(value);
	if (value < pow(1024.0, 2))
		return (spec ~ " KiB").format(value / 1024.0);
	if (value < pow(1024.0, 3))
		return (spec ~ " MiB").format(value / pow(1024.0, 2));
	if (value < pow(1024.0, 4))
		return (spec ~ " GiB").format(value / pow(1024.0, 3));
	if (value < pow(1024.0, 5))
		return (spec ~ " TiB").format(value / pow(1024.0, 4));
	if (value < pow(1024.0, 6))
		return (spec ~ " PiB").format(value / pow(1024.0, 5));
	if (value < pow(1024.0, 7))
		return (spec ~ " EiB").format(value / pow(1024.0, 6));
	if (value < pow(1024.0, 8))
		return (spec ~ " ZiB").format(value / pow(1024.0, 7));
	
	return (spec ~ " YiB").format(value / pow(1024.0, 8));
}

enum ANSIColor
{
	reset        = "0",
	white        = "1",
	red          = "31",
	green        = "32",
	yellow       = "33",
	brightRed    = "1;31",
	brightGreen  = "1;32",
	brightYellow = "1;33"
}

string ANSIEncode(in ANSIColor color) nothrow pure @safe
{
	return "\x1b[" ~ color ~ "m";
}

string ansiFormat(in string str, in ANSIColor color) nothrow pure @safe
{
	return color.ANSIEncode ~ str ~ ANSIColor.reset.ANSIEncode;
}

version (unittest)
{
	public import std.stdio : write, writeln;
	import std.traits : fullyQualifiedName;
	
	template writeTest(alias T)
	{
		enum writeTest = "write(\"\x1b[1m" ~ fullyQualifiedName!T ~ "...\x1b[0m\");
			scope(failure) writeln(\"\x1b[1;31m failure!\x1b[0m\");
			scope(success) writeln(\"\x1b[1;32m success =)\x1b[0m\");";
	}
	
	template notTested(alias T)
	{
		enum notTested = "writeln(\"\x1b[1m" ~ fullyQualifiedName!T ~ "... \x1b[33mnot tested.\x1b[0m\");";
	}
}

