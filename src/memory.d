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
	@property ref T tail() @nogc nothrow pure @safe
	{
		return _tail._payload;
	}
	
	/**
	 * Returns: The last element of the list.
	 */
	@property ref T head() @nogc nothrow pure @safe
	{
		return _head._payload;
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
	
	// Implementing the List as a range.
	
	/**
	 * Returns: Current element of the list.
	 */
	ref T front() @nogc nothrow pure @safe
	{
		return _current._payload;
	}
	
	/**
	 * Move the curent pointer to the next element.
	 */
	void popFront() @nogc nothrow pure @safe
	{
		if (_current !is null)
			_current = _current._next;
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

