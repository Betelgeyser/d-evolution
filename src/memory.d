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

immutable poolSize = 128 * 2^^20; /// Size of a newly allocated block. Defaults to 128 MiB. This number is purely random,
                                  /// probably it will need some optimization later.

debug(memory)
{
	import std.exception : assumeWontThrow;
	import std.stdio     : writeln;
	
	/**
	 * Dirty way to supress some errors in debug builds that apperantly should not happen, like failing to compile
	 * $(D_KEYWORD nothrow) function inspite it should be ignored with $(D_KEYWORD debug)(memory).
	 */
	void writeLog(T...)(T args, string file = __FILE__, int line = __LINE__) @nogc nothrow pure @safe
	{
		debug(memory) assumeWontThrow(writeln(file, " ", line, "\t", args));
	}
}

/**
 * Simple double-linked list.
 *
 * It is tuned to the specific task of memory menegement. E. g. it lacks any search and relies more on iteration through it.
 */
struct List(T)
{
	this(this) @nogc nothrow pure @safe
	{
		_current = _tail; // We need to reset the _current pointer on copy for iterators
	}
	
	private
	{
		/**
		 * Structure that wraps list element with poitners to its neighbors.
		 */
		struct Node(T)
		{
			@disable this();
			@disable this(this);
			
			private
			{
				T       _payload;     /// The data itself.
				Node!T* _prev = null; /// Pointer to the previous node.
				Node!T* _next = null; /// Pointer to the next node.
			}
		}
		
		Node!T* _tail    = null; /// Pointer to the first element of the list.
		Node!T* _head    = null; /// Pointer to the last element of the list.
		Node!T* _current = null; /// Pointer to the currently iterated element of the list.
		
		size_t _length; /// Number of elements in the list.
	}
	
	/**
	 * Returns: Number of elements in the list.
	 */
	@property size_t length() const @nogc nothrow pure @safe
	{
		return _length;
	}
	
	/**
	 * Returns: Pointer to the first element of the list.
	 */
	@property T* tail() @nogc nothrow pure @safe
	{
		if (_tail !is null)
			return &_tail._payload;
		else
			return null;
	}
	
	/**
	 * Returns: Pointer to the last element of the list.
	 */
	@property T* head() @nogc nothrow pure @safe
	{
		if (_head !is null)
			return &_head._payload;
		else
			return null;
	}
	
	/**
	 * Returns: Pointer to the element previous to the current one.
	 */
	@property T* prev() @nogc nothrow pure @safe
	{
		if (_current._prev !is null)
			return &_current._prev._payload;
		else
			return null;
	}
	
	/**
	 * Returns: Pointer to the element next to the current one.
	 */
	@property T* next() @nogc nothrow pure @safe
	{
		if (_current._next !is null)
			return &_current._next._payload;
		else
			return null;
	}
	
	/**
	 * Append a new element at the end of the list. The current pointer is then set to a new head.
	 *
	 * Params:
	 *     value = Value of a the new element.
	 */
	void pushFront(T value) @nogc nothrow
	{
		auto newHead = cast(Node!T*)malloc((Node!T).sizeof);
		
		newHead._payload = value;
		newHead._next    = null;
		newHead._prev    = _head;
		
		if (_head !is null)
			_head._next = newHead;
		
		if (_tail is null)
			_tail = newHead;
		
		if (_current is null)
			_current = newHead;
		
		_head = newHead;
		
		++_length;
	}
	
	/**
	 * Insert a new element into the list after the current one. The new element will be set as a current one.
	 *
	 * Params:
	 *     value = Value of a the new element.
	 */
	void insertAfter(T value) @nogc nothrow
	{
		auto newNode = cast(Node!T*)malloc((Node!T).sizeof);
		
		newNode._payload = value;
		
		if (_current is null)
		{
			newNode._prev = null;
			newNode._next = null;
		}
		else
		{
			newNode._prev = _current;
			newNode._next = _current._next;
			
			if (_current._next is null)
				_head = newNode;
			else
				_current._next._prev = newNode;
			
			_current._next = newNode;
		}
		
		_current = newNode;
		
		if (_tail is null)
			_tail = newNode;
		
		if (_head is null)
			_head = newNode;
		
		++_length;
	}
	
	/**
	 * Remove an element before the current one.
	 */
	void removeBefore() @nogc nothrow
	{
		_remove(_current._prev);
	}
	
	/**
	 * Remove an element after the current one.
	 */
	void removeAfter() @nogc nothrow
	{
		_remove(_current._next);
	}
	
	/**
	 * Remove an element $(D_PARAM node) from the list.
	 */
	private void _remove(Node!T* node) @nogc nothrow
	{
		if (node !is null)
		{
			if (node == _tail)
				_tail = node._next;
			
			if (node == _head)
				_head = node._prev;
			
			if (node._prev !is null)
				node._prev._next = node._next;
			
			if (node._next !is null)
				node._next._prev = node._prev;
			
			free(node);
			
			--_length;
		}
	}
	
	/**
	 * Returns: $(D_KEYWORD true) if there are no more elements avaliable in the list.
	 */
	bool empty() const @nogc nothrow pure @safe
	{
		return _current is null;
	}
	
	/**
	 * For foreach.
	 *
	 * Foreach implemented through ranges will result in somewhat different behaviour. Fetching an element will move
	 * `_current` pointer to the next node so the fetched and `_current` node would be different. This behaviour is even more
	 * inconsistent when there is only one node in the list. In this case fetched and `_current` nodes would be same.
	 *
	 * That way, opApply ensures coherence between fetched and `_current` elements when foreach loop is used.
	 */
	auto opApply(scope int delegate(ref T) @nogc nothrow dg) @nogc nothrow
	{
		int result = 0;
		
		_current = _tail;
		while (!empty)
		{
			result = dg(_current._payload);
			if (result)
				break;
			
			_current = _current._next;
		}
		_current = _tail;
		
		return result;
	}
}

///
unittest
{
	mixin(writeTest!List);
	
	struct S
	{
		int   i;
		float f;
	}
	
	List!S list;
	
	with (list)
	{
		pushFront( S(1, 1f) );
		assert (_current._payload == S(1, 1f));
		assert (_current          == _tail);
		assert (_current          == _head);
		assert (_current._prev    is null);
		assert (_current._next    is null);
		assert (_length           == 1);
		
		pushFront( S(3, 1f) );
		assert (_current       == _tail);
		assert (_tail._payload == S(1, 1f));
		assert (_tail._next    == _head);
		assert (_head._payload == S(3, 1f));
		assert (_head._prev    == _tail);
		assert (_head._next    is null);
		assert (_length        == 2);
		
		insertAfter( S(2, 1f) );
		assert (_current._payload == S(2, 1f));
		assert (_current          != _tail);
		assert (_current          != _head);
		assert (_current._prev    == _tail);
		assert (_current._next    == _head);
		assert (_tail._next       == _current);
		assert (_head._prev       == _current);
		assert (_length           == 3);
		
		int i = 1;
		foreach (el; list)
		{
			assert (el == S(i, 1f));
			++i;
		}
		
		// Try to remove this foreach and foreach by reference will suddenly fail. DMD64 D Compiler v2.079.0
		foreach (el; list)
			el.f = 2f;
		foreach (el; list)
			assert (el.f == 1f, "foreach by value changed an element in the list.");
		
		foreach (ref el; list)
			el.f = 2f;
		foreach (el; list)
			assert (el.f == 2f, "foreach by reference failed to change an element in the list.");
		
		foreach (ref el; list)
		{
			if (el == S(2, 1f))
			{
				removeBefore();
				removeAfter();
				assert (_current._payload == S(2, 1f));
				assert (_current          == _tail);
				assert (_current          == _head);
				assert (_current._prev    is null);
				assert (_current._next    is null);
				assert (_length           == 1);
			}
		}
	}
}

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
	
	this(void* ptr, in size_t size, in bool isAllocated = false, void* prev = null, void* next = null) @nogc nothrow pure @safe
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
	@property void* ptr() @nogc nothrow pure @safe
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
	@property void* prev() @nogc nothrow pure @safe
	{
		return _prev;
	}
	
	/**
	 * Returns: Pointer of (not 'to') the next adjucent block.
	 */
	@property void* next() @nogc nothrow pure @safe
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
	 * Allocates memory in the block.
	 *
	 * Params:
	 *     size = Allocated bytes.
	 *
	 * Returns:
	 *     Pointer to the allocated memory.
	 */
	void* allocate(in size_t size) @nogc nothrow pure @safe
	{
		debug(memory) assert (isFree, "Cannot allocate in already allocated block.");
		debug(memory) assert (
			_size >= size,
			assumeWontThrow(
				"%s %d\tBlock cannot allocate %d bytes. Only %d is available."
				.format(__FILE__, __LINE__, size, _size)
		));
		
		_isAllocated = true;
		_size        = size;
		
		return _ptr;
	}
	
	/**
	 * Free block.
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
	Block[] split(in size_t size) nothrow pure
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
	static Block merge(Block leftBlock, Block rightBlock)
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
 * Memory pool.
 *
 * It is a sufficiently large area of memory which is allocated at once. Any further allocations are done from this region
 * which improves performance by reducing the number of cudaMallocManaged calls.
 *
 * Every pool is devided into blocks which may be allocated or free. Whenever new allocation is performed, pool looks for
 * the first free block of the sufficient size and splits this block into two parts: allocated and free (if any free space
 * is left).
 */
struct Pool
{
	/**
	 * Params:
	 *     size = Size of the pool.
	 */
	this(in size_t size) @nogc nothrow
	{
		_size = size;
		
		auto block = Block(size);
		_blocks.pushFront(block);
	}
	
	private
	{
		immutable size_t _size;   /// Size of the pool.
		List!Block       _blocks; /// List of the pool's blocks.
	}
	
	/**
	 * Allocates memory in the pool.
	 *
	 * Params:
	 *     size = Size in bytes to allocate.
	 *
	 * Returns:
	 *     Pointer to the allocated memory. If there were no sufficient block found in the pool, then $(D_KEYWORD null)
	 *     is returned.
	 */
	void* allocate(in size_t size) @nogc nothrow
	{
		debug(memory) size_t i = 0;
		foreach (ref block; _blocks)
		{
			debug(memory) writeLog(
				"Scanning ", i, " block at ", block.ptr, ". ",
				"It is ", block.isFree ? "free" : "allocated", ", its size is ", block.size, " bytes"
			);
			if (block.isFree && block.size >= size)
			{
				debug(memory) writeLog("This block is sufficient. Allocating");
				
				auto freeBlock = Block(block.ptr + size, block.size - size, false);
				_blocks.insertAfter(freeBlock);
				
				debug(memory) writeLog("New free block address is ", freeBlock.ptr);
				
				return block.allocate(size);
			}
			debug(memory) ++i;
		}
		
		return null;
	}
	
	/**
	 * Free allocated block.
	 *
	 * Params:
	 *     ptr = Pointer to a block being freed.
	 *
	 * Returns:
	 *     $(D_KEYWORD true) if the block were successefuly freed. If the block pointed by this pointer were not found
	 *     in the pool and were not freed, then $(D_KEYWORD false) is returned. 
	 */
	bool free(void* ptr) @nogc nothrow
	{
		debug(memory) size_t i = 0;
		foreach (ref block; _blocks)
		{
			debug(memory) writeLog("Searching in the ", i, " block");
			if (block.ptr == ptr)
			{
				debug(memory) writeLog("Ptr ", ptr, " is found");
				block.free();
				
				if (_blocks.prev !is null && _blocks.prev.isFree)
				{
					debug(memory) writeLog("Previous block ", _blocks.prev.ptr, " is free. Merging");
					
					block = Block(_blocks.prev.ptr, _blocks.prev.size + block.size, false);
					_blocks.removeBefore();
				}
				
				if (_blocks.next !is null && _blocks.next.isFree)
				{
					debug(memory) writeLog("Next block ", _blocks.next.ptr, " is free. Merging");
					
					block = Block(block.ptr, block.size + _blocks.next.size, false);
					_blocks.removeAfter();
				}
				
				return true;
			}
			debug(memory) ++i;
		}
		
		return false;
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
	
	private
	{
		List!Pool _pools; /// List of avaliable pools.
		Block[void*] _allocatedBlocks; /// AA of all existing memory blocks.
		Block[void*]      _freeBlocks; /// AA of free memory blocks.
	}
	
	/**
	 * Allocate an array.
	 *
	 * Params:
	 *     T (template) = The type of elements to allocate. Must be numerical POD.
	 *     items = Number of items to allocate.
	 *
	 * Returns:
	 *     Allocated array.
	 */
	T[] allocate(T)(in size_t items) nothrow
		if (isNumeric!T)
	{
		auto size = items * T.sizeof;
		
		if (size > poolSize)
			throw new Error("Allocating %d bytes, but maximum pool size is %d bytes.".format(size, poolSize));
		
		debug(memory) writeLog("Allocating ", size, " bytes.");
		
		return cast(T[])_firstFit(size)[0 .. size];
	}
	
	/**
	 * Free an allocated array.
	 *
	 * Params:
	 *     array = Array to free.
	 */
	void free(T)(ref T[] array) @nogc nothrow
	{
		debug(memory) writeLog("Freeing ptr ", array.ptr);
		debug(memory) size_t i = 0;
		foreach (ref pool; _pools)
		{
			debug(memory) writeLog("Searching in the ", i, " pool");
			
			if (pool.free(array.ptr))
			{
				array.destroy();
				return;
			}
			debug(memory) ++i;
		}
		debug(memory) assert (false, "%s %d\tUnable to free %s, memory violation".format(__FILE__, __LINE__, array.ptr));
	}
	
	T[] allocate2(T)(in size_t items) nothrow
		if (isNumeric!T)
	{
		if (items == 0)
			return null;
		
		size_t size = items * T.sizeof;
		
		foreach (freeBlock; _freeBlocks.byValue())
		{
			if (freeBlock.size >= size)
			{
				auto newBlocks = freeBlock.split(size);
				
				_allocatedBlocks[newBlocks[0].ptr] = newBlocks[0];
				_allicatedBlocks[newBlocks[0].ptr].allocate();
				
				if (newBlocks.lenght == 2)
					_freeBlocks[newBlocks[1].ptr] = newBlocks[1].ptr;
				
				_freeBlocks.remove(freeBlock.ptr);
				
				return cast(T[])newBlocks[0].ptr[0 .. size];
			}
		}
		
		return null;
	}
	
	void free2(T)(ref T[] array) nothrow
	{
		auto block = _allocatedBlocks[array.ptr];
		
		if (block.prev in _freeBlocks)
		{
			auto prevBlock = _freeBlocks[block.prev];
			
			newPtr = prevBlock.ptr;
			newSize += prevBlock.size;
			newPrev = prevBlock.prev;
			
			_freeBlocks.remove(block.prev);
		}
		
		if (block.next in _freeBlocks)
		{
			auto nextBlock = _freeBlocks[block.next];
			
			newSize += nextBlock.size;
			newNext = nextBlock.next;
			
			_freeBlocks.remove(block.next);
		}
		
		_freeBlocks[newPtr] = Block(newPtr, newSize, false, newPrev, newNext);
		_allocatedBlocks.remove(block.ptr);
		
		array = null;
	}
	
	private
	{
		/**
		 * Allocating using the firts fit stratagy.
		 *
		 * Params:
		 *     size = Size in bytes to allocate.
		 *
		 * Returns:
		 *     Pointer to the allocated memory.
		 *
		 * See_Also:
		 *     $(LINK http://www.memorymanagement.org/mmref/alloc.html#first-fit)
		 */
		void* _firstFit(in size_t size) @nogc nothrow
		{
			debug(memory) size_t i = 0;
			foreach (ref pool; _pools)
			{
				debug(memory) writeLog("Searching a sufficient block in the ", i, " pool");
				
				auto ptr = pool.allocate(size);
				if (ptr !is null)
					return ptr;
				
				debug(memory) ++i;
			}
			
			debug(memory) writeLog("No sufficient pool was found. Creating a new one");
			
			_addPool(poolSize);
			
			return _pools.head.allocate(size);
		}
		
		/**
		 * Create a new pool and add it to the list.
		 *
		 * Params:
		 *     size = Size of a new pool in bytes.
		 */
		void _addPool(in size_t size) @nogc nothrow
		{
			auto pool = Pool(poolSize);
			_pools.pushFront(pool);
		}
	}
}

///
unittest
{
	// This is a simplified unittest and it cannot detect every possible error, but is OK for now.
	// Use `-debug memory` switch to enable full memory logging in case of errors.
	
	mixin(writeTest!UnifiedMemoryManager);
	UnifiedMemoryManager manager;
	
	float[][] a;
	
	a ~= manager.allocate!float(15728640);
	assert (manager._pools.length == 1); // Created 1st pool
	assert (manager._pools.head._blocks.length == 2); // 1st block in the pool is splited into two new blocks:
	                                                  // allocated and free
	assert (a[0].length == 15728640);
	assert (a[0][$ - 1] == a[0][$ - 1]); // Ckeck memory accessability
	
	a ~= manager.allocate!float(15728640);
	assert (manager._pools.length == 1); // As there are enough free space in the 1st pool, no new pool is created
	assert (manager._pools.head._blocks.length == 3); // New block is allocated from the free one
	assert (a[1].length == 15728640);
	assert (a[1][$ - 1] == a[1][$ - 1]);
	
	a ~= manager.allocate!float(15728640);
	assert (manager._pools.length == 2); // As there is not enough free space in the 1st pool, a new one is created
	assert (manager._pools.tail._blocks.length == 3); // The 1st pool is unaffected
	assert (manager._pools.head._blocks.length == 2); // A new block is created in the 2nd pool
	assert (a[2].length == 15728640);
	assert (a[2][$ - 1] == a[2][$ - 1]);
	
	manager.free(a[0]);
	assert (manager._pools.length == 2); // Pools are never freed right now
	assert (manager._pools.tail._blocks.length == 3); // Freed block is not adjacent to any other free block so it can't be merged
	assert (manager._pools.head._blocks.length == 2);
	assert (a[0].ptr    == null); // The destroyed array must not be accessable any more
	assert (a[0].length == 0);
	
	manager.free(a[1]);
	assert (manager._pools.length == 2); // Pools are never freed right now
	assert (manager._pools.tail._blocks.length == 1); // Freed block was adjacent to free blocks from the both sides, so they
	                                                  // have been merged together
	assert (manager._pools.head._blocks.length == 2);
	assert (a[1].ptr    == null);
	assert (a[1].length == 0);
	
	a[0] = manager.allocate!float(15728640);
	assert (manager._pools.length == 2);
	assert (manager._pools.tail._blocks.length == 2); // As now there is enough free space in the 1st pool, a new array is
	                                                  // allocated there
	assert (manager._pools.head._blocks.length == 2);
	assert (a[0].length == 15728640);
	assert (a[0][$ - 1] == a[0][$ - 1]); // As we allocated a[0] one more time, it must be accessable again
}

