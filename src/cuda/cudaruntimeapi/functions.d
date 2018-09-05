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
 * Higher-level wrappers around CUDA runtime API.
 *
 * This module allows to use CUDA runtime API calls in a D-style using features like templates and error checking through asserts.
 */
module cuda.cudaruntimeapi.functions;

import cuda.common;
import cuda.cudaruntimeapi.types;
static import cudart = cuda.cudaruntimeapi.exp;

void cudaMalloc(T)(ref T* devPtr, ulong nitems) nothrow @nogc
{
	void* tmp;
	enforceCudart(cudart.cudaMalloc(&tmp, nitems * T.sizeof));
	devPtr = cast(T*)tmp;
}

deprecated
void cudaMallocManaged(T)(ref T* devPtr, ulong nitems, uint flags = cudaMemAttachGlobal) nothrow @nogc
{
	void* tmp;
	enforceCudart(cudart.cudaMallocManaged(&tmp, nitems * T.sizeof, flags));
	devPtr = cast(T*)tmp;
}

void cudaMallocManaged(T)(ref T[] devPtr, ulong nitems, uint flags = cudaMemAttachGlobal) nothrow @nogc
{
	void* tmp;
	enforceCudart(cudart.cudaMallocManaged(&tmp, nitems * T.sizeof, flags));
	devPtr = (cast(T*)tmp)[0 .. nitems];
}

deprecated
void cudaFree(void* devPtr) nothrow @nogc
{
	enforceCudart(cudart.cudaFree(devPtr));
}

void cudaFree(T)(ref T[] devPtr) nothrow @nogc
{
	enforceCudart(cudart.cudaFree(devPtr.ptr));
	devPtr.destroy();
}

void cudaMemcpy(void* dst, const(void)* src, size_t count, cudaMemcpyKind kind) nothrow @nogc
{
	enforceCudart(cudart.cudaMemcpy(dst, src, count, kind));
}

void cudaDeviceSynchronize() nothrow @nogc
{
	enforceCudart(cudart.cudaDeviceSynchronize());
}

int cudaGetDeviceCount() @nogc nothrow
{
	int result;
	enforceCudart(cudart.cudaGetDeviceCount(&result));
	return result;
}

void cudaSetDevice(int device)
{
	enforceCudart(cudart.cudaSetDevice(device));
}

/**
 * Utility wrapper to enforce error check for cuda functions.
 */
package void enforceCudart(cudaError_t error) pure nothrow @safe @nogc
{
	assert (error == cudaError_t.cudaSuccess, error.toString);
}

