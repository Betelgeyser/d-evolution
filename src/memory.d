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
module memory;

// Standard D modules
import core.stdc.stdlib : free, malloc;

import common;

debug(memory)
{
	import std.exception : assumeWontThrow;
	import std.string    : format;
}

immutable poolSize = 128 * 2^^20; /// Size of a newlly allocated block. Defaults to 128 MiB. This number is purely random,
                                  /// probably it will need some optimization later.

debug(memory)
{
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
 * It is tuned to the specific task of memory menegement. E.g. it lacks any search and relies more on iteration through it.
 *
 * TODO: Not thread safe!
 */
struct List(T)
{
	this(this) @nogc nothrow pure @safe
	{
		_current = _tail; // For proper functionality we need to reset the _current pointer on copy
	}
	
	private
	{
		/**
		 * Structure that wraps list element with poitners to its neighbors.
		 *
		 * TODO: Probably nulldable will be better here, but I didn't manage to make it work. Nevertheless, tail(), head(),
		 * etc. are not lvalues, so might be no difference.
		 */
		struct Node(T)
		{
			alias _payload this;
			
			@disable this();
			@disable this(this);
			
			private
			{
				T       _payload;     /// The data itself.
				Node!T* _prev = null; /// Pointer to the previous node.
				Node!T* _next = null; /// Pointer to the nex node.
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
	 * Returns: The first element of the list.
	 */
	@property T* tail() @nogc nothrow pure @safe
	{
		if (_tail !is null)
			return &_tail._payload;
		else
			return null;
	}
	
	/**
	 * Returns: The last element of the list.
	 */
	@property T* head() @nogc nothrow pure @safe
	{
		if (_head !is null)
			return &_head._payload;
		else
			return null;
	}
	
	/**
	 * Returns: The element previous to the current one.
	 */
	@property T* prev() @nogc nothrow pure @safe
	{
		if (_current._prev !is null)
			return &_current._prev._payload;
		else
			return null;
	}
	
	/**
	 * Returns: The element next to the current one.
	 */
	@property T* next() @nogc nothrow pure @safe
	{
		if (_current._next !is null)
			return &_current._next._payload;
		else
			return null;
	}
	
	/**
	 * Append a new element at the end of the list.
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
	 * Remove an element &(D_PARAM node) from the list.
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
				node._prev._next = _current._next;
			
			if (node._next !is null)
				node._next._prev = _current._prev;
			
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
private struct Block
{
	@disable this();
	
	/**
	 * Params:
	 *     ptr = Root pointer of the block.
	 *     size = Size of the block in bytes.
	 *     isAllocated = Has block already been allocated?
	 */
	this(in size_t size) @nogc nothrow
	{
		_ptr         = cudaMallocManaged(size);
		_size        = size;
		_isAllocated = false;
	}
	
	/// ditto
	this(UnifiedPointer ptr, in size_t size, in bool isAllocated) @nogc nothrow pure @safe
	{
		_ptr         = ptr;
		_size        = size;
		_isAllocated = isAllocated;
	}
	
	private
	{
		UnifiedPointer _ptr; /// Pointer to the block's memory.
		
		size_t _size;        /// Size of the block.
		bool   _isAllocated; /// `true` if the block has been allocated.
	}
	
	/**
	 * Returns: Pointer to the memory block.
	 */
	@property UnifiedPointer ptr() @nogc nothrow pure @safe
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
	 * Returns: `true` if the block is free.
	 */
	@property bool isFree() const @nogc nothrow pure @safe
	{
		return !_isAllocated;
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
	UnifiedPointer allocate(in size_t size) @nogc nothrow pure @safe
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
		debug(memory) writeLog(_ptr, " is freed");
	}
}

/**
 * Memory pool.
 *
 * It is a sufficiently large area of memory which is allocated at once. Any further allocations are done from this region
 * which improves performance by reducing the number of cudaMallocManaged calls.
 *
 * Every pool is devided into blocks which may be allocated or free. Whenever new allocation is performed, pool looks for
 * the first free block of the sufficient size and splits this block into two parts: allocated and free (if some free space
 * is left).
 */
private struct Pool
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
		List!Block   _blocks; /// List of the pool's blocks.
		const size_t _size;   /// Size of the pool.
	}
	
	/**
	 * Allocates memory in the pool.
	 *
	 * Params:
	 *     size = Allocated bytes.
	 *
	 * Returns:
	 *     Pointer to the allocated memory.
	 */
	UnifiedPointer allocate(in size_t size) @nogc nothrow
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
				
				debug(memory) writeLog("New free block adress is ", freeBlock.ptr);
				
				return block.allocate(size);
			}
			debug(memory) ++i;
		}
		
		return UnifiedPointer(null);
	}
	
	bool free(UnifiedPointer ptr) @nogc nothrow
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

alias UMM = UnifiedMemoryManager;

/**
 * Cuda unified memory manager. The main purpose of the manager is speeding up slow cudaMallocManaged. It also provides more
 * type-safe system for cuda pointers.
 *
 * The problem with cudaMalloc and cudaMallocManaged is calls to these functions are slow. Requested size
 * does not matters, calls to these functions are slow themselvs. That doesn't generally affect performance much,
 * but if there are a lot of calls to the cuda mallocs (which is the case with this program), performance drops drastically.
 * As a sugesttion, the issue could be caused by switching between kernel and user modes every time the function is called.
 * Interestingly enough, there is no such an issue with regular malloc.
 *
 * As a solution, UnifiedMemoryManager allocates large blocks of memory at once (which are called pools) and allocates
 * returns small parts of these pools. This increases an amount of memory used by the application, but minimizes number
 * of direct cudaMallocManaged calls.
 *
 * Allocated memory can be freed and used for further allocations, but in current implementation will never be returned to
 * an OS.
 *
 * Currently it uses only first-fit allocation strategy, as one of the fastest and easiest to implement.
 */
struct UnifiedMemoryManager
{
	@disable this(this);
	
	private
	{
		List!Pool _pools; /// List of avaliable pools.
	}
	
	/**
	 * Allocate array.
	 */
	UnifiedArray!T allocate(T)(in size_t items) @nogc nothrow
	{
		auto size = items * T.sizeof;
		
		debug(memory) writeLog("Allocating ", size, " bytes");
		
		return UnifiedArray!T(_firstFit(size), items);
	}
	
	/**
	 * Free an allocated array.
	 *
	 * NOTE: Accessing pointed area after the array has been freed is undefined behaviour.
	 */
	void free(T)(UnifiedArray!T array) @nogc nothrow
	{
		debug(memory) writeLog("Freeing ptr ", array.ptr);
		debug(memory) size_t i = 0;
		foreach (ref pool; _pools)
		{
			debug(memory) writeLog("Searching in the ", i, " pool");
			
			if (pool.free(array.ptr))
				return;
			debug(memory) ++i;
		}
		debug(memory) assert (false, "%s %d\tUnable to free %s, memory violation".format(__FILE__, __LINE__, array.ptr));
	}
	
	private
	{
		/**
		 * Allocating using the firts fit stratagy.
		 */
		auto _firstFit(in size_t size) @nogc nothrow
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
		 */
		void _addPool(in size_t size) @nogc nothrow
		{
			auto pool = Pool(poolSize);
			_pools.pushFront(pool);
		}
	}
}


