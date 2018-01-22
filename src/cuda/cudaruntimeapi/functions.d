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
module cuda.cudaruntimeapi.functions;

import cuda.cudaruntimeapi.types;

private extern(C) cudaError_t cudaMalloc(void** devPtr, size_t size) nothrow @nogc;

extern(C) cudaError_t cudaFree(void* devPtr) nothrow @nogc;
extern(C) cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nothrow @nogc;
extern(C) cudaError_t cudaGetLastError() nothrow @nogc;
extern(C) const(char)* cudaGetErrorString(cudaError_t error) nothrow @nogc;

extern(C) cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem = 0, cudaStream_t stream = null) nothrow @nogc;


/**
 * Higher level wrapper around cudaMalloc.
 */
cudaError_t cudaMalloc(T)(ref T* devPtr, ulong nitems) nothrow @nogc
{
	void* tmp;
	cudaError_t err = cudaMalloc(&tmp, nitems * T.sizeof);
	devPtr = cast(T*)tmp;
	return err;
}

