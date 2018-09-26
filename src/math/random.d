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
module math.random;

// C modules
import core.stdc.stdlib;

// Standard D modules
import std.exception : enforce;
import std.string    : format;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;
public import cuda.curand.types : curandRngType_t;

// DNN modules
import common;


version (unittest)
{
	RandomPool randomPool;
	private CurandGenerator curandGenerator;
	
	static this()
	{
		curandGenerator = CurandGenerator(curandRngType_t.PSEUDO_DEFAULT);
		randomPool      = RandomPool(curandGenerator);
	}
	
	static ~this()
	{
		randomPool.freeMem();
		curandGenerator.destroy;
	}
}

/**
 * Pool of random bits.
 *
 * NOT THREAD SAFE!!!
 *
 * As cuRAND achieves maximun performance generating big amounts of data, this structure will generate a pool of uniform
 * random bits. If all numbers in the pool has been used or there is not enought values in the pool to return,
 * then a new pool is generated.
 *
 * TODO: Thread safety. Pool returns a reference to a memory. If another thread requests new values and pool regenerates,
 * then previously returned reference will be pointing to wrong, newly generated values.
 */
struct RandomPool
{
	private
	{
		uint[]  _values; /// Cached random bits.
		size_t* _index;  /// Index of the values that will be returned next.
		CurandGenerator _generator; /// Curand generator.
	}
	
	invariant
	{
		if (_values.length && _index)           // TODO: Dirty hack. As invariant is called right after `freeMem()`,
			assert (*_index <= _values.length); // this line will cause crash.
	}
	
	@property size_t length() const pure nothrow @safe @nogc
	{
		return _values.length;
	}
	
	/**
	 * Setup the pool and generate new bits.
	 *
	 * Params:
	 *     generator = Curand pseudorandom number generator.
	 *     size = Pool size in bytes. Defaults to 2GiB.
	 */
	this(in curandRngType_t rngType, in ulong seed = 0, in size_t size = 536_870_912) nothrow
	out
	{
		assert (_values.length == size);
	}
	body
	{
		if (size < 1)
			throw new Error("RandomPool must have at least 1 value, got %d.".format(size));
		
		scope(failure) freeMem();
		
		_generator = CurandGenerator(rngType, seed);
		
		_index  = cast(typeof(_index))malloc(_index.sizeof);
		*_index = 0;
		
		cudaMallocManaged(_values, size);
		
		_generator.generate(_values.ptr, length);
		cudaDeviceSynchronize();
	}
	
	/**
	 * Free memory.
	 *
	 * For the reason how D works with structs memory freeing is moved from destructor to
	 * the the distinct function. Either allocating structs on stack or in heap or both
	 * causes spontaneous destructors calls. Apparently structs are not intended
	 * to be used with dynamic memory, probably it should be implemented as a class.  
	 */
	void freeMem() nothrow
	{
		if (_index)
		{
			free(_index);
			_index = null;
		}
		
		_generator.freeMem();
		
		if (_values.length)
			cudaFree(_values);
	}
	
	/**
	 * Get `count` new random values from the pool.
	 *
	 * If there is not enought values in the pool, than new values will be generated.
	 *
	 * Params:
	 *     count = How many values to return.
	 *
	 * Returns:
	 *     a slice of random numbers that has not been used.
	 */
	uint[] opCall(in size_t count) nothrow
	{
		if (count > length)
			throw new Error("RandomPool has %d values, but %d requested.".format(length, count));
		
		if (count > _available)
			regenerate();
		
		return _values[*_index .. *_index += count];
	}
	
	private
	{
		/**
		 * Number of values that has not been used.
		 */
		@property size_t _available() const pure nothrow @safe @nogc
		{
			return length - *_index;
		}
		
		/**
		 * Generates new values and resets _index.
		 */
		void regenerate() nothrow @nogc
		{
			*_index = 0;
			_generator.generate(_values.ptr, length);
			cudaDeviceSynchronize();
		}
	}
}

///
unittest
{
	mixin(writeTest!RandomPool);
	
	immutable size = 1_000;
	
	// Initialize cuRAND generator.
	auto generator = CurandGenerator(curandRngType_t.PSEUDO_DEFAULT);
	generator.setPseudoRandomGeneratorSeed(0);
	scope(exit) generator.destroy;
	
	// Initialize pool
	auto pool = RandomPool(generator, size);
	scope(exit) pool.freeMem();
	
	// There is a chance of getting two equal numbers in a row, but chance is low
	assert (pool(1)[0] != pool(1)[0]);
	
	// Ensure pool regenerates its values
	assert (pool(size)[0] != pool(size)[0]);
}

