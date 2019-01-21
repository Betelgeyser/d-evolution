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
 *
 * This module contains memory management things.
 */
module memory; // TODO: All contents of this module are non thread-safe!

/*
 * TODO: Use associative arrays (hashmaps) to free block at O(n) and then append them list of free blocks.
 * One more option is to use self balancing binary search tree for free blocks to speed up searching of a sufficient free block.
 */

// Standard D modules
import core.stdc.stdlib : free, malloc;
import std.conv         : to;
import std.string       : format;
import std.traits       : isNumeric;
import std.typecons     : Nullable, nullable, Tuple, tuple;

import common;

import cuda.cudaruntimeapi : cudaMallocManaged;

UnifiedMemoryManager UMM;

private:

immutable blockSize = 128 * 2^^20; /// Size of a newly allocated block. Defaults to 128 MiB. This number is purely random,
                                          /// probably it will need some optimization later.

/**
 * Continuous block of memory. It is the minimum unit of allocation.
 *
 * In fact, it is just a wrapper around a pointer storing some additional information, such as avaliable size of the pointed
 * memory and whether it has already been allocated.
 */
struct Block
{
	@disable this();
	
	/**
	 * Allocate a new block of a size $(D_PARAM size).
	 *
	 * This constructor does actual memory allocation.
	 *
	 * Params:
	 *     size = Size of the block in bytes.
	 */
	this(in size_t size) @nogc nothrow
	{
		cudaMallocManaged(_ptr, size);
		_size        = size;
		_isAllocated = false;
	}
	
	/**
	 * Create a new block.
	 *
	 * Params:
	 *     ptr = Pointer to the block.
	 *     size = Size of the new block.
	 *     isAllocated = Has this block already been allocated or not.
	 */
	this(void* ptr, in size_t size, in bool isAllocated = false) @nogc nothrow pure
	{
		_ptr         = ptr;
		_size        = size;
		_isAllocated = isAllocated;
	}
	
	this(void* ptr, in size_t size, in bool isAllocated = false, void* prev = null, void* next = null) @nogc nothrow pure
	{
		_ptr         = ptr;
		_size        = size;
		_isAllocated = isAllocated;
		_prev        = prev;
		_next        = next;
	}
	
	private
	{
		void*  _ptr;         /// Pointer to the block's memory.
		size_t _size;        /// Size of the block.
		bool   _isAllocated; /// $(D_KEYWORD true) if the block is allocated.
		void*  _prev = null; /// Pointer to the previous adjucent block.
		void*  _next = null; /// Pointer to the next adjucent block.
	}
	
	invariant
	{
		assert (_size > 0);
		assert (_ptr !is null);
	}
	
	/**
	 * Returns: Pointer to the memory block.
	 */
	@property void* ptr() @nogc nothrow pure
	{
		return _ptr;
	}
	
	/**
	 * Returns: Size of the block in bytes.
	 */
	@property size_t size() const @nogc nothrow pure @safe
	{
		return _size;
	}
	
	/**
	 * Returns: Pointer of (not 'to') the previous adjucent block.
	 */
	@property void* prev() @nogc nothrow pure
	{
		return _prev;
	}
	
	/**
	 * Returns: Pointer of (not 'to') the next adjucent block.
	 */
	@property void* next() @nogc nothrow pure
	{
		return _next;
	}
	
	/**
	 * Returns: $(D_KEYWORD true) if the block is free.
	 */
	@property bool isFree() const @nogc nothrow pure @safe
	{
		return !_isAllocated;
	}
	
	/**
	 * Returns: $(D_KEYWORD true) if the block is allocated.
	 */
	@property bool isAllocated() const @nogc nothrow pure @safe
	{
		return _isAllocated;
	}
	
	/**
	 * Mark the block as allocated.
	 */
	void allocate() @nogc nothrow pure @safe
	{
		_isAllocated = true;
	}
	
	/**
	 * Mark the block as free.
	 *
	 * The block is not returned to the OS, it is just been marked as free so it can be allocated further.
	 */
	void free() @nogc nothrow pure @safe
	{
		_isAllocated = false;
	}
	
	/**
	 * Split a block into two blocks.
	 *
	 * Although this function splits the block into two blocks, if $(D_PARAM size) is set to 0 or equals to the block size,
	 * only one block is returned, that is $(D_KEYWORD this) block itself.
	 *
	 * This function is useful as it automatically manages `_prev` and `_next` pointers of splited blocks.
	 *
	 * Params:
	 *     Size = Size of the left block.
	 *
	 * Returns:
	 *     Array of one or two blocks. The firts one is of $(D_PARAM size) size, the second one is of the remaining size.
	 */
	Block[] split(in size_t size) nothrow
	out(result)
	{
		assert (result.length == 1 || result.length == 2);
	}
	body
	{
		if (size > _size)
			throw new Error("Block split into %d bytes, while only %d is avaliable.".format(size, _size));
		
		if (size == 0)
			return [this];
		
		if (size < _size)
		{
			void*  nextPtr  = _ptr + size;
			size_t nextSize = _size - size;
			
			return [
				Block(_ptr,    size,     _isAllocated, _prev, nextPtr),
				Block(nextPtr, nextSize, _isAllocated, _ptr,  _next)
			];
		}
		
		return [this];
	}
	
	///
	unittest
	{
		mixin(writeTest!split);
		
		auto block = Block(100);
		
		assert (block.split(0).length          == 1);
		assert (block.split(block.size).length == 1);
		
		assert (block.split(0)[0]          == block);
		assert (block.split(block.size)[0] == block);
		
		immutable splitSize = 40;
		auto splitBlocks = block.split(splitSize);
		
		assert (splitBlocks.length == 2);
		
		assert (splitBlocks[0].ptr == block.ptr);
		assert (splitBlocks[1].ptr == block.ptr + splitSize);
		
		assert (splitBlocks[0].size == splitSize);
		assert (splitBlocks[1].size == block.size - splitSize);
		
		assert (splitBlocks[0].prev is null);
		assert (splitBlocks[1].prev == splitBlocks[0].ptr);
		
		assert (splitBlocks[0].next == splitBlocks[1].ptr);
		assert (splitBlocks[1].next is null);
	}
	
	/// TODO: Error control.
	static Block merge(Block leftBlock, Block rightBlock) nothrow
	{
		return Block(
			leftBlock.ptr,
			leftBlock.size + rightBlock.size,
			leftBlock.isAllocated,
			leftBlock.prev,
			rightBlock.next
		);
	}
	
	///
	unittest
	{
		mixin(writeTest!merge);
		
		auto block = Block(100);
		auto splitBlocks = block.split(60);
		
		assert (merge(splitBlocks[0], splitBlocks[1]) == block);
	}
}

/**
 * Cuda unified memory manager. The main purpose of the manager is speeding up slow cudaMallocManaged.
 *
 * The problem with cudaMalloc and cudaMallocManaged is that calls to these functions are slow. Requested size
 * does not matter, calls to these functions are slow themselves. That doesn't generally affect performance much,
 * but if there are a lot of calls to the cuda mallocs (which is the case with this program), performance drops drastically.
 * As a sugestion, the issue could be caused by switching between kernel and user modes every time the function is called.
 * Interestingly enough, there is no such an issue with regular malloc.
 *
 * As a solution, UnifiedMemoryManager allocates large blocks of memory at once (which are called pools) and returns small
 * parts of these pools. This increases an amount of memory used by the application, but minimizes number of direct
 * cudaMallocManaged calls.
 *
 * Allocated memory can be freed and used for further allocations, but in current implementation will never be returned to
 * the OS.
 *
 * Currently it uses only the first-fit allocation strategy, as one of the fastest and easiest to implement.
 */
struct UnifiedMemoryManager
{
	@disable this(this);
	
	@property size_t capacity()
	{
		return _capacity;
	}

	// TODO: Freeing memory
	
	private
	{
		Block[void*] _allocatedBlocks; /// AA of all existing memory blocks.
		Block[void*] _freeBlocks;      /// AA of free memory blocks.
		
		size_t _capacity;
	}
	
	T[] allocate(T)(in size_t items) nothrow
		if (isNumeric!T)
	{
		if (items == 0)
			return null;
		
		T[] result = _allocate!T(items);
		
		if (result is null)
		{
			this._extend();
			result = _allocate!T(items);
		}
		
		if (result is null)
			throw new Error("Memory allocation error.");
		
		return result;
	}
	
	void free(T)(ref T[] array) nothrow
	{
		if (array is null)
			return;
		
		auto block = _allocatedBlocks[array.ptr];
		block.free();
		
		_allocatedBlocks.remove(block.ptr);
		_freeBlocks[block.ptr] = block;
		
		if (block.next in _freeBlocks)
		{
			auto nextBlock = _freeBlocks[block.next];
			
			_freeBlocks.remove(block.next);
			_freeBlocks[block.ptr] = Block.merge(block, nextBlock);
			_freeBlocks[block.ptr].free();
			
			block = _freeBlocks[block.ptr];
		}
		
		if (block.prev in _freeBlocks)
		{
			auto prevBlock = _freeBlocks[block.prev];
			
			_freeBlocks.remove(block.ptr);
			_freeBlocks[block.prev] = Block.merge(prevBlock, block);
			_freeBlocks[block.prev].free();
		}
		
		array = null;
	}
	
	private
	{
		T[] _allocate(T)(in size_t items) nothrow
		{
			size_t size = T.sizeof * items;
			
			foreach (freeBlock; _freeBlocks.byValue())
			{
				if (freeBlock.size >= size)
				{
					auto newBlocks = freeBlock.split(size);
					
					_allocatedBlocks[newBlocks[0].ptr] = newBlocks[0];
					_allocatedBlocks[newBlocks[0].ptr].allocate();
					
					if (newBlocks.length == 2)
						_freeBlocks[newBlocks[1].ptr] = newBlocks[1];
					
					_freeBlocks.remove(freeBlock.ptr);
					
					return cast(T[])newBlocks[0].ptr[0 .. size];
				}
			}
			
			return null;
		}
		
		void _extend() nothrow
		{
			auto block = Block(blockSize);
			_freeBlocks[block.ptr] = block;
			_capacity += blockSize;
		}
	}
}

///
unittest
{
	mixin(notTested!UnifiedMemoryManager);
//	UnifiedMemoryManager manager;
//	
//	auto a = manager.allocate!float(0);
//	assert (a is null);
//	assert (a.length == 0);
//	
//	manager.free(a); // Must not crash
//	
//	import std.stdio;
//	writeln();
//	
//	auto a1 = manager.allocate!float(15_000_000); // 4KiB of memory
//	writeln("15_000_000 float allocated");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
//	
//	auto a2 = manager.allocate!float(15_000_000);
//	writeln("60_000_000 byte allocated");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
//	
//	auto a3 = manager.allocate!float(15);
//	writeln("15 float allocated");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
//	
//	auto a4 = manager.allocate!float(15_000_000);
//	writeln("15_000_000 float allocated");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
//	
//	manager.free(a2);
//	assert (a2 is null);
//	writeln("2 is freed");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
//	
//	manager.free(a3);
//	assert (a3 is null);
//	writeln("3 is freed");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
//	
//	manager.free(a1);
//	manager.free(a4);
//	writeln("all is freed");
//	writeln(manager._allocatedBlocks);
//	writeln(manager._freeBlocks);
}

