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

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.curand;

// DNN modules
import common;

/**
 * Pool of uniformly distributed random numbers in range (0; 1].
 *
 * As cuRAND achieves maximun performance generating big amounts of data, this structure will generate pool of random numbers
 * and return them on request. If all numbers in the pool was used or there is not enought values in a pool, then new pool
 * will be generated.
 */
struct RandomPool
{
	immutable ulong size;  /// Pool size. 
	
	private
	{
		float[] _values; /// Random values cache.
		ulong   _index;  /// Last returned pointer to a pool element.
		curandGenerator _generator; /// Generator regenerates numbers as the pool runs out of them.
	}
	
	invariant
	{
		assert (_values.length == size);
		assert (_index <= size);
	}
	
	/**
	 * Setup a pool and generate new numbers.
	 *
	 * Params:
	 *     generator = Curand pseudorandom number generator.
	 *     size = Pool size. The maximun amount of generated values to store. Defaults to the size of 2GiB values.
	 */
	this(curandGenerator generator, in uint size = 536_870_912) nothrow @nogc
	in
	{
		assert (size >= 1);
	}
	body
	{
		this.size = size;
		_generator = generator;
		
		cudaMallocManaged(_values, size);
		scope(failure) freeMem();
		
		_generator.generateUniform(_values.ptr, size);
		cudaDeviceSynchronize();
	}
	
	/**
	 * Get `count` new random values from the pool.
	 *
	 * If there is not enought values in a pool, than new values will be generated.
	 *
	 * Params:
	 *     count = How many values to return.
	 *
	 * Returns:
	 *     Pointer to random numbers that were not used.
	 */
	const(float)[] opCall(in ulong count) nothrow @nogc
	in
	{
		assert (count >= 1 && count <= size);
	}
	body
	{
		if (count > available)
			regenerate();
		
		const(float)[] result = _values[_index .. _index + count];
		_index += count;
		return result;
	}
	
	/**
	 * Free memory.
	 *
	 * For the reason how D works with structs memory freeing moved from destructor to
	 * the the distinct function. Either allocating structs on stack or in heap or both
	 * causes spontaneous destructors calls. Apparently structs are not intended
	 * to be used with dynamic memory, probably it should be implemented as a class.  
	 */
	void freeMem() nothrow @nogc
	{
		cudaFree(_values);
	}
	
	private
	{
		/**
		 * Number of values that has not been used.
		 */
		@property ulong available() const pure nothrow @safe @nogc
		{
			return size - _index;
		}
		
		/**
		 * Generates new values and resets _index.
		 */
		void regenerate() nothrow @nogc
		{
			_index = 0;
			_generator.generateUniform(_values.ptr, size);
			cudaDeviceSynchronize();
		}
	}
}

///
unittest
{
	mixin(writetest!RandomPool);
	
	import std.math : approxEqual;
	immutable accuracy = 0.000_001;
	
	immutable size = 1_000;
	
	// Initialize cuRAND generator.
	auto generator = curandGenerator(curandRngType_t.PSEUDO_DEFAULT);
	generator.setPseudoRandomGeneratorSeed(0);
	scope(exit) generator.destroy;
	
	// Initialize pool
	auto p = RandomPool(generator, size);
	scope(exit) p.freeMem();
	
	// There is a chance of getting two equal floats in a row, but it's virtually impossible
	assert ( !approxEqual(
			p(1)[0],
			p(1)[0],
			accuracy
	));
	
	p(size); // force pool to regenerate
	
	// Ensure pool regenerates its values
	assert ( !approxEqual(
			p(size / 2 + 1)[0],
			p(size / 2 + 1)[0],
			accuracy
	));
}

