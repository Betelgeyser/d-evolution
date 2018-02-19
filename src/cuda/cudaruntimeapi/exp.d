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
 * Cuda runtime API exported functions.
 */
module cuda.cudaruntimeapi.exp;

import cuda.cudaruntimeapi.types;

extern(C) package nothrow @nogc:
	cudaError_t cudaMalloc(void** devPtr, size_t size);
	cudaError_t cudaMallocManaged(void** devPtr, size_t size, uint flags = cudaMemAttachGlobal); 
	cudaError_t cudaFree(void* devPtr);
	cudaError_t cudaMemcpy(void* dst, const(void)* src, size_t count, cudaMemcpyKind kind);
	cudaError_t cudaDeviceSynchronize();
