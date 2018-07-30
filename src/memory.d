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

immutable poolSize = 128 * 2^^20; /// Size of a newlly allocated block. Defaults to 128 MiB. This number is purely random,
                                   /// probably it will need some optimization later.

/**
 * Simple double-linked list.
 */
struct List(T)
{
	private
	{
		/**
		 * Structure that wraps list element with poitners to its neighbors.
		 */
		struct Node(T)
		{
			private
			{
				T       _payload;     /// The data itself.
				Node!T* _prev = null; /// Pointer to the previous node.
				Node!T* _next = null; /// Pointer to the nex node.
			}
			
			alias _payload this;
			
			@disable this();
			@disable this(this);
		}
		
		Node!T* _tail    = null; /// Pointer to the first element of the list.
		Node!T* _head    = null; /// Pointer to the last element of the list.
		Node!T* _current = null; /// Pointer to the currently iterated element of the list.
	}
	
	this(this) @nogc nothrow pure @safe
	{
		_current = _tail; // For proper functionality we need to reset the _current pointer
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
	}
	
	void removeBefore() @nogc nothrow
	{
		_remove(_current._prev);
	}
	
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
		}
	}
	
	/**
	 * Returns: true if there are no more nodes avaliable in the list.
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
	int opApply(int delegate(ref T) dg)
	{
		int result = 0;
		
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
//	mixin(writeTest!List);
//	
}

/**
 * Continuous block of memory.
 *
 * In fact, it is just a wrapper around a pointer storing some additional information, such as avaliable size of the pointed
 * memory and whether it has already been allocated.
 */
private struct Block
{
	private
	{
		UnifiedPointer _ptr; /// Pointer to the block's memory.
		
		size_t _size;        /// Size of the block.
		bool   _isAllocated; /// `true` if the block has been allocated.
	}
	
	/**
	 * Returns: Pointer to the memory block.
	 */
	@property UnifiedPointer ptr() @nogc nothrow @safe
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
	
	/**
	 * Allocates memory in the block.
	 *
	 * Params:
	 *     size = Allocated bytes.
	 *
	 * Returns:
	 *     Pointer to the allocated memory.
	 */
	UnifiedPointer allocate(in size_t size) pure @safe
	{
		assert (isFree, "Cannot allocate in already allocated block.");
		assert (_size >= size, "Block cannot allocate %d bytes. Only %d is available.".format(size, _size));
		
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
		debug(memory) assumeWontThrow(writeln(__FILE__, " ", __LINE__, "\t", _ptr, " is freed"));
	}
}

/**
 * Memory pool.
 */
private struct Pool
{
	private
	{
		List!Block   _blocks; /// List of the pool's blocks.
		const size_t _size;   /// Size of the pool.
	}
	
	@disable this();
	
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
	
	/**
	 * Allocates memory in the pool.
	 *
	 * Params:
	 *     size = Allocated bytes.
	 *
	 * Returns:
	 *     Pointer to the allocated memory.
	 */
	UnifiedPointer allocate(in size_t size)
	{
		debug(memory) size_t i = 0;
		foreach (ref block; _blocks)
		{
			debug(memory) write(__FILE__, " ", __LINE__, "\tScanning ", i, " block at ", block.ptr, ". ");
			debug(memory) write("It is ", block.isFree ? "free" : "allocated", ", its size is ", block.size, " bytes. This block is ");
			
			if (block.isFree && block.size >= size)
			{
				debug(memory) writeln("sufficient. Allocating");
				auto freeBlock = Block(block.ptr + size, block.size - size, false);
				_blocks.insertAfter(freeBlock);
				debug(memory) writeln(__FILE__, " ", __LINE__, "\tNew free block adress is ", freeBlock.ptr);
				
				return block.allocate(size);
			}
			debug(memory) writeln("insufficient");
			debug(memory) ++i;
		}
		
		return UnifiedPointer(null);
	}
	
	bool free(UnifiedPointer ptr)
	{
		debug(memory) size_t i = 0;
		foreach (ref block; _blocks)
		{
			debug(memory) writeln(__FILE__, " ", __LINE__, "\tSearching in the ", i, " block");
			if (block.ptr == ptr)
			{
				debug(memory) writeln(__FILE__, " ", __LINE__, "\tPtr ", ptr, " is found");
				block.free();
				
				if (_blocks.prev !is null && _blocks.prev.isFree)
				{
					debug(memory) writeln(__FILE__, " ", __LINE__, "\tPrevious block ", _blocks.prev.ptr, " is free. Merging");
					
					block = Block(_blocks.prev.ptr, _blocks.prev.size + block.size, false);
					_blocks.removeBefore();
				}
				
				if (_blocks.next !is null && _blocks.next.isFree)
				{
					debug(memory) writeln(__FILE__, " ", __LINE__, "\tNext block ", _blocks.next.ptr, " is free. Merging");
					
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
 * Cuda unified memory manager.
 */
struct UnifiedMemoryManager
{
	private
	{
		List!Pool _pools; /// List of avaliable pools.
	}
	
	@disable this(this);
	
	/**
	 * Allocate array.
	 */
	UnifiedArray!T allocate(T)(in size_t items)
	{
		auto size = items * T.sizeof;
		
		debug(memory) writeln(__FILE__, " ", __LINE__, "\tAllocating ", size, " bytes");
		
		return UnifiedArray!T(_firstFit(size), items);
	}
	
	/**
	 * Free an allocated array.
	 */
	void free(T)(UnifiedArray!T array)
	{
		debug(memory) writeln(__FILE__, " ", __LINE__, "\tFreeing ptr ", array.ptr);
		debug(memory) size_t i = 0;
		foreach (ref pool; _pools)
		{
			debug(memory) writeln(__FILE__, " ", __LINE__, "\tSearching in the ", i, " pool");
			
			if (pool.free(array.ptr))
				return;
			debug(memory) ++i;
		}
		assert (false, "%s %d\tUnable to free %s, memory violation".format(__FILE__, __LINE__, array.ptr));
	}
	
	private
	{
		/**
		 * Allocating using the firts fit stratagy.
		 */
		auto _firstFit(in size_t size)
		{
			debug(memory) size_t i = 0;
			foreach (ref pool; _pools)
			{
				debug(memory) writeln(__FILE__, " ", __LINE__, "\tSearching a sufficient block in the ", i, " pool");
				
				auto ptr = pool.allocate(size);
				if (ptr !is null)
					return ptr;
				
				debug(memory) ++i;
			}
			
			debug(memory) writeln(__FILE__, " ", __LINE__, "\tNo sufficient pool was found. Creating a new one");
			
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


