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
 * Higher-level wrappers around CUDA runtime API.
 *
 * This module allows to use CUDA runtime API calls in a D-style using features like templates and error checking through asserts.
 */
module cuda.cudaruntimeapi.functions;

import cuda.common;
import cuda.cudaruntimeapi.types;
static import cudart = cuda.cudaruntimeapi.exp;

void cudaMalloc(T)(ref T* devPtr, ulong nitems) @nogc nothrow
{
	void* tmp;
	enforceCudart(cudart.cudaMalloc(&tmp, nitems * T.sizeof));
	devPtr = cast(T*)tmp;
}

void cudaMallocManaged(T)(ref T* devPtr, ulong nitems, uint flags = cudaMemAttachGlobal) @nogc nothrow
{
	void* tmp;
	enforceCudart(cudart.cudaMallocManaged(&tmp, nitems * T.sizeof, flags));
	devPtr = cast(T*)tmp;
}

void cudaMallocManaged(T)(ref T[] devPtr, ulong nitems, uint flags = cudaMemAttachGlobal) @nogc nothrow
{
	void* tmp;
	enforceCudart(cudart.cudaMallocManaged(&tmp, nitems * T.sizeof, flags));
	devPtr = (cast(T*)tmp)[0 .. nitems];
}

/**
 * Safe wrapper around cudaMallocManaged.
 *
 * This function is trusted because it returns an array rather than a raw pointer.
 * The length of the array is exactly the number of the allocated items. The array itself
 * provides a safer interface to the allocated memory as it performs bounds check.
 *
 * The value of the array can be reassigned, its length can be changed and that will cause
 * a GC allocation. In that case cuda memory will not be accessible any more. This behaviour is better
 * to be avoided, but it should not affect memory safety.
 */
T[] cudaMallocManaged(T)(ulong nitems, uint flags = cudaMemAttachGlobal) @nogc nothrow pure @trusted
{
	void* tmp;
	enforceCudart(cudart.cudaMallocManaged(&tmp, nitems * T.sizeof, flags));
	return (cast(T*)tmp)[0 .. nitems];
}

void cudaFree(void* devPtr) @nogc nothrow
{
	enforceCudart(cudart.cudaFree(devPtr));
}

void cudaFree(T)(ref T[] devPtr) @nogc nothrow
{
	enforceCudart(cudart.cudaFree(devPtr.ptr));
	devPtr.destroy();
}

void cudaMemcpy(void* dst, const(void)* src, size_t count, cudaMemcpyKind kind) @nogc nothrow pure
{
	enforceCudart(cudart.cudaMemcpy(dst, src, count, kind));
}

void cudaDeviceSynchronize() @nogc nothrow
{
	enforceCudart(cudart.cudaDeviceSynchronize());
}

int cudaGetDeviceCount() @nogc nothrow pure
{
	int result;
	enforceCudart(cudart.cudaGetDeviceCount(&result));
	return result;
}

void cudaSetDevice(int device) @nogc nothrow
{
	enforceCudart(cudart.cudaSetDevice(device));
}

/**
 * Utility wrapper to enforce error check for cuda functions.
 */
package void enforceCudart(cudaError_t error) @nogc nothrow pure @safe
{
	if (error != cudaError_t.cudaSuccess)
		throw new Error(error.toString);
}

