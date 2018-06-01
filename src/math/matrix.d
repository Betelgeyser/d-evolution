/**
 * Copyright © 2017 - 2018 Sergei Iurevich Filippov, All Rights Reserved.
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
module math.matrix;

// Standard D modules
import std.range : ElementType;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;

// DNN modules
import common;

version (unittest)
{
	import std.algorithm : each, equal;
	import std.math      : approxEqual;
	
	private cublasHandle_t cublasHandle;
	
	static this()
	{
		cublasCreate(cublasHandle);
	}
	
	static ~this()
	{
		cublasDestroy(cublasHandle);
	}
}


/**
 * Convenient struct to handle cuBLAS matricies.
 *
 * It wraps pointer to allocated memory of values with some additional properties of a matrix, such as rows and columns
 * numbers. All linear algebra method are implemented on GPU.
 *
 * As it implements cuBLAS matrix, it is column-major ordered.
 */
struct Matrix
{
	alias values this;
	
	// Pointer should not be reassignable.
	float[] values; /// A pointer to an allocated memory.
	
	private
	{
		uint _rows; /// Number of rows.
		uint _cols; /// Number of columns.
	}
	
	invariant
	{
		assert (_rows >= 1);
		assert (_cols >= 1);
	}
	
	/**
	 * Number of rows.
	 */
	@property uint rows() const @nogc nothrow pure @safe
	{
		return _rows;
	}
	
	/**
	 * Returns: The number of columns.
	 */
	@property uint cols() const @nogc nothrow pure @safe
	{
		return _cols;
	}
	
	/**
	 * The length of the matrix.
	 *
	 * Returns: The number of elements.
	 */
	@property size_t length() const @nogc nothrow pure @safe
	{
		return values.length;
	}
	
	/**
	 * Returns: The size of the matrix in bytes.
	 */
	@property size_t size() const @nogc nothrow pure @safe
	{
		return length * ElementType!(typeof(values)).sizeof;
	}
	
	/**
	 * Create a matrix and allocate memory for it.
	 *
	 * Default values are not initialized. If a cuRAND generator is passed,
	 * values are randomly generated on GPU.
	 *
	 * Params:
	 *     rows = Number of rows.
	 *     cols = Number of columns.
	 *     values = Array of values.
	 */
	this(in uint rows, in uint cols) @nogc nothrow
	in
	{
		assert (rows >= 1);
		assert (cols >= 1);
	}
	out
	{
		// Many std.algorithm's higher order functions on ranges will pop elements from `values` decreasing its lenght.
		// That makes it imposible to place this check in the invariant section.
		assert (values.length == _rows * _cols);
	}
	body
	{
		scope(failure) freeMem();
		
		_rows = rows;
		_cols = cols;
		
		cudaMallocManaged(values, _rows * _cols);
	}
	
	/// ditto
	this(in uint rows, in uint cols, inout(float)[] values) inout @nogc nothrow @safe
	in
	{
		assert (values.length == cols * rows);
	}
	body
	{
		_rows = rows;
		_cols = cols;
		this.values = values;
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		immutable rows = 2;
		immutable cols = 3;
		
		auto matrix = Matrix(rows, cols);
		scope(exit) matrix.freeMem();
		
		cudaDeviceSynchronize();
		
		assert (matrix.rows == rows);
		assert (matrix.cols == cols);
		
		// Check memory accessebility
		assert (matrix[0]     == matrix[0]);
		assert (matrix[$ - 1] == matrix[$ - 1]);
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
		cudaFree(values);
	}
	
	/**
	 * Returns a column slice of a matrix from i column to `j - 1`.
	 *
	 * This function acts like a normal slice but a unit of slicing is a single column rather than single value. The resulting
	 * matrix points to a part of the original matrix and does not copy it.
	 */
	inout(Matrix) colSlice(in uint i, in uint j) inout nothrow @safe @nogc
	in
	{
		assert (i < j, "No columns is returned.");
		assert (j <= _cols, "Column index is out of range.");
	}
	body
	{
		return inout Matrix(_rows, j - i, values[i * _rows .. j * _rows]);
	}
	
	///
	unittest
	{
		mixin(writeTest!colSlice);
		
		immutable size = 10;
		immutable from =  2;
		immutable to   =  5;
		
		auto matrix = Matrix(size, size);
		scope(exit) matrix.freeMem();
		
		auto copy = matrix.colSlice(from, to);
		
		assert (copy.cols == to - from);
		assert (copy.rows == matrix.rows);
		
		foreach (i, c; copy)
			assert (c == matrix[from * matrix.rows + i]);
	}
}

/**
 * Simpified version of gemm function.
 *
 * Performs matrix multiplication C = A * B.
 *
 * Params:
 *     A = The first matrix.
 *     B = The second matrix.
 *     C = Output matrix.
 *     cublasHandle = Cublas handle.
 */
void gemm(in Matrix A, in Matrix B, ref Matrix C, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (A.cols == B.rows);
	assert (A.rows == C.rows);
	assert (B.cols == C.cols);
}
body
{
	immutable float alpha = 1;
	immutable float beta  = 0;
	
	cublasSgemm(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
		A.rows, B.cols, A.cols,
		&alpha,
		A.ptr, A.rows,
		B.ptr, B.rows,
		&beta,
		C.ptr, C.rows
	);
}

///
unittest
{
	mixin(writeTest!gemm);
	
	immutable n = 7;
	immutable k = 3;
	immutable m = 5;
	
	auto A = Matrix(n, k);
	scope(exit) A.freeMem();
	
	auto B = Matrix(k, m);
	scope(exit) B.freeMem();
	
	auto C = Matrix(n, m);
	scope(exit) C.freeMem();
	
	A.each!"a = i";
	B.each!"a = i";
	
	gemm(A, B, C, cublasHandle);
	cudaDeviceSynchronize();
	
	// cuBLAS is column-major
	immutable float[] result = [
		 35,  38,  41,  44,  47,  50,  53,
		 98, 110, 122, 134, 146, 158, 170,
		161, 182, 203, 224, 245, 266, 287,
		224, 254, 284, 314, 344, 374, 404,
		287, 326, 365, 404, 443, 482, 521
	];
	
	assert (equal!approxEqual(C.values, result));
}

/**
 * Simpified version of geam function.
 *
 * Performs matrix addition. C = αA + βB.
 *
 * Params:
 *     alpha = α constant multiplier.
 *     A = The first matrix.
 *     beta = β constant multiplier.
 *     B = The second matrix.
 *     C = Output matrix.
 *     cublasHandle = Cublas handle.
 */
void geam(in float alpha, in Matrix A, in float beta, in Matrix B, ref Matrix C, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert ( (A.cols == B.cols) && (A.cols == C.cols) );
	assert ( (A.rows == B.rows) && (A.rows == C.rows) );
}
body
{
	cublasSgeam(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N,
		C.rows, C.cols,
		&alpha,
		A.ptr, A.rows,
		&beta,
		B.ptr, B.rows,
		C.ptr, C.rows
	);
}

///
unittest
{
	mixin(writeTest!geam);
	
	immutable size = 10;
	
	auto A = Matrix(size, size);
	scope(exit) A.freeMem();
	
	auto B = Matrix(size, size);
	scope(exit) B.freeMem();
	
	auto C = Matrix(size, size);
	scope(exit) C.freeMem();
	
	A.each!"a = i";
	B.each!"a = i";
	
	geam(1, A, 2, B, C, cublasHandle);
	cudaDeviceSynchronize();
	
	foreach (i, c; C)
		assert (approxEqual(c, 3 * i));
}

/**
 * Simpified version of geam function.
 *
 * Performs matrix transposition.
 *
 * Params:
 *     A = Matrix to transponse.
 *     C = Resulting matrix.
 *     cublasHandle = Cublas handle.
 */
void transpose(in Matrix A, ref Matrix C, cublasHandle_t cublasHandle) nothrow @nogc
in
{
	assert (A.rows == C.cols);
	assert (A.cols == C.rows);
}
body
{
	immutable float alpha = 1;
	immutable float beta  = 0;
	
	cublasSgeam(
		cublasHandle,
		cublasOperation_t.CUBLAS_OP_T, cublasOperation_t.CUBLAS_OP_N,
		A.cols, A.rows,
		&alpha,
		A.ptr, A.rows,
		&beta,
		A.ptr, A.rows,
		C.ptr, C.rows
	);
}

///
unittest
{
	mixin(writeTest!transpose);
	
	immutable m = 5;
	immutable n = 3;
	
	auto A = Matrix(m, n);
	scope(exit) A.freeMem();
	
	auto C = Matrix(n, m);
	scope(exit) C.freeMem();
	
	A.each!"a = i";
	
	transpose(A, C, cublasHandle);
	cudaDeviceSynchronize();
	
	// cuBLAS is column-major
	immutable float[] result = [
		0, 5, 10,
		1, 6, 11,
		2, 7, 12,
		3, 8, 13,
		4, 9, 14
	];
	
	assert (equal!approxEqual(C.values, result));
}

