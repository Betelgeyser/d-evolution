/**
 * Copyright © 2017 - 2019 Sergei Iurevich Filippov, All Rights Reserved.
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
import std.algorithm    : each, count;
import std.conv         : to;
import std.csv          : csvReader;
import std.exception    : enforce;
import std.range        : ElementType;
import std.regex        : ctRegex, matchFirst;
import std.string       : format;

// CUDA modules
import cuda.cudaruntimeapi;
import cuda.cublas;

import memory;

// DNN modules
import common;

version (unittest)
{
	import std.algorithm : equal;
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

import core.exception : RangeError;


/**
 * Convenient struct to handle cuBLAS matricies.
 *
 * It wraps pointer to allocated memory of values with some additional properties of a matrix, such as rows and columns
 * numbers. All linear algebra method are implemented on GPU.
 *
 * As it implements cuBLAS matrix, it is column-major ordered. This means, that a matrix
 *
 * <math xmlns = "http://www.w3.org/1998/Math/MathML">
 *     <mrow><mo>[</mo><mtable>
 *         <mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>
 *         <mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>
 *         <mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>
 *     </mtable><mo>]</mo></mrow>
 * </math>
 *
 * will be stored as [1, 1, 1, 2, 2, 2] in memory.
 */
struct Matrix
{
	alias values this;
	
	// TODO: Pointer should not be reassignable.
	float[] values; /// A pointer to an allocated memory.
	
	{
	}
	
	/**
	 * Create a matrix and allocate memory for it.
	 *
	 * Default values are not initialized.
	 *
	 * Params:
	 *     rows = Number of rows.
	 *     cols = Number of columns.
	 */
	this(in uint rows, in uint cols) nothrow
	out
	{
		// Many std.algorithm's higher order functions on ranges will pop elements from `values` decreasing its lenght.
		// That makes it imposible to place this check in the invariant section.
		assert (values.length == _rows * _cols);
	}
	body
	{
		scope(failure) freeMem();
		
		if (rows < 1 || cols < 1)
			throw new Error("Wrong matrix size.");
		
		_rows = rows;
		_cols = cols;
		
		version(UMM) values = UMM.allocate!float(_rows * _cols);
		else cudaMallocManaged(values, _rows * _cols);
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
	 * Create a matrix and allocate memory for it from csv data.
	 *
	 * Params:
	 *     csv = Coma separated values.
	 */
	this(in string csv)
	{
		scope(failure) freeMem();
		
		_rows = csv.count("\n").to!uint;
		
		auto firstLineCTR = ctRegex!("^.*\n");
		string firstLine = matchFirst(csv, firstLineCTR)[0];
		_cols = firstLine.count(",").to!uint + 1;
		
		version(UMM) values = UMM.allocate!float(_rows * _cols);
		else cudaMallocManaged(values, _rows * _cols);
		
		size_t i = 0;
		foreach (record; csv.csvReader!float)
		{
			size_t j = 0;
			foreach (value; record)
				this[i, j++] = value;
			
			++i;
		}
	}
	
	///
	unittest
	{
		mixin(writeTest!__ctor);
		
		auto A = Matrix("1.2\n");
		scope(exit) A.freeMem();
		
		assert (A.rows == 1 && A.cols == 1);
		assert (approxEqual(A.values[0], 1.2));
		
		auto B = Matrix("1.2,3.4,5.6\n");
		scope(exit) B.freeMem();
		
		assert (B.rows == 1 && B.cols == 3);
		assert (equal!approxEqual(B.values, [1.2, 3.4, 5.6]));
		
		auto C = Matrix("1.2\n3.4\n5.6\n");
		scope(exit) C.freeMem();
		
		assert (C.rows == 3 && C.cols == 1);
		assert (equal!approxEqual(C.values, [1.2, 3.4, 5.6]));
		
		auto D = Matrix("1,2,3\n1,2,3\n");
		scope(exit) D.freeMem();
		
		assert (D.rows == 2 && D.cols == 3);
		assert (equal!approxEqual(D.values, [1, 1, 2, 2, 3, 3])); // Remember, cuBLAS is column-major
	}
	
	size_t opDollar() const @nogc nothrow pure @safe
	{
		return values.length;
	}
	
	float opIndex(in size_t i) const @nogc nothrow pure @safe
	{
		return values[i];
	}
	
	float opIndexAssign(in float value, in size_t i) @nogc nothrow pure @safe
	{
		return values[i] = value;
	}
	
	float opIndex(in size_t i, in size_t j) const @nogc nothrow pure @safe
	{
		if (i >= _rows || j >= _cols)
			throw new Error("Matrix range violation.");
		
		return values[i + j * _rows];
	}
	
	float opIndexAssign(in float value, in size_t i, in size_t j) @nogc nothrow pure @safe
	{
		if (i >= _rows || j >= _cols)
			throw new Error("Matrix range violation.");
		
		return values[i + j * _rows] = value;
	}
	
	/**
	 * Returns: The number of rows.
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
	 * Returns: Pointer to the values array.
	 */
	@property inout(float*) ptr() inout @nogc nothrow pure
	{
		return _values.ptr;
	}
	
	/**
	 * Wraps an array into a matrix.
	 *
	 * Or, could be used to create a new matrix from a part of another one.
	 *
	 * Params:
	 *     rows = Number of rows.
	 *     cols = Number of columns.
	 *     origin = Array to create matrix from.
	 */
	private this(in uint rows, in uint cols, inout(float[]) origin) inout nothrow pure @safe
	in
	{
		assert (origin.length == cols * rows, "Matrix size is %dx%d, but got %d elements.".format(rows, cols, origin.length));
		assert (rows >= 1 && cols >= 1, "Matrix size must be at least 1x1, got %dx%d".format(rows, cols));
	}
	body
	{
		_rows = rows;
		_cols = cols;
		
		values = origin;
	
	private
	{
		uint _rows; /// Number of rows.
		uint _cols; /// Number of columns.
	}
	
	invariant
	{
		assert (_rows >= 1 && _cols >= 1, "Matrix size must be at least 1x1, got %dx%d".format(_rows, _cols));
	}
	
	/**
	 * Free memory.
	 *
	 * For the reason how D works with structs memory freeing moved from destructor to
	 * the the distinct function. Either allocating structs on stack or in heap or both
	 * causes spontaneous destructors calls. Apparently structs are not intended
	 * to be used with dynamic memory, probably it should be implemented as a class.  
	 */
	void freeMem() nothrow
	{
		version(UMM) UMM.free(values);
		else cudaFree(values);
	}
	
	/**
	 * Deep copy.
	 *
	 * Params:
	 *     src = Matrix to copy.
	 *     dst = Destination matrix.
	 */
	static void copy(in Matrix src, Matrix dst) nothrow pure @safe
	{
		if (src.rows != dst.rows || src.cols != dst.cols)
			throw new Error(
				"Source and destination matrices have different sizes %dx%d and %dx%d."
				.format(src.rows, src.cols, dst.rows, dst.cols)
			);
		
		src.values.each!((i, x) => dst.values[i] = x);
	}
	
	/**
	 * Returns a column slice of a matrix from $(D_PARAM i) column to $(D_PARAM j) - 1.
	 *
	 * This function acts like a normal slice but a unit of slicing is a single column rather than single value. The resulting
	 * matrix points to a part of the original matrix and does not copy it.
	 */
	inout(Matrix) colSlice(in uint i, in uint j) inout nothrow pure @safe
	{
		if (j > _cols)
			throw new RangeError("Matrix size is %dx%d, but %d column is indexed.".format(_rows, _cols, j));
		
		if (i >= j)
			throw new RangeError("Invalid range [%d, %d].".format(i, j));
		
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
 * A convenient wrapper around cublasSgemm.
 *
 * Performs the matrix-matrix multiplication C = op(A)op(B) + C.
 *
 * Despite this function is similar to original cublasSgemm function it provides a better interface.
 * Instead of taking raw pointers data and cryptic dimentions parameters it takes matrices and
 * figures out dimentions on its own. Also this function provides better control over dimentions
 * of the matrises.
 *
 * Params:
 *     A = The first matrix.
 *     transA = If $(D_KEYWORD true) then the matrix A is transposed.
 *     B = The second matrix.
 *     transB = If $(D_KEYWORD true) then the matrix B is transposed.
 *     C = Output matrix.
 *     cublasHandle = Cublas handle.
 *
 * See_also:
 *     For reference and implementation details see
 *     $(LINK https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)
 */
void gemm(in Matrix A, in bool transA, in Matrix B, in bool transB, ref Matrix C, cublasHandle_t cublasHandle) nothrow
{
	cublasOperation_t opA = transA ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
	cublasOperation_t opB = transB ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
	
	int m = transA ? A.cols : A.rows;
	int n = transB ? B.rows : B.cols;
	int k = transA ? A.rows : A.cols;
	
	int lda = A.rows;
	int ldb = B.rows;
	int ldc = C.rows;
	
	if (C.rows != m || C.cols != n || k != (transB ? B.cols : B.rows))
		throw new Error("Invalid matrix-matrix multiplication.");
	
	immutable float alpha = 1;
	immutable float beta  = 0;
	
	cublasSgemm(
		cublasHandle,
		opA, opB,
		m, n, k,
		&alpha,
		A.ptr, lda,
		B.ptr, ldb,
		&beta,
		C.ptr, ldc
	);
}

///
unittest
{
	mixin(writeTest!gemm);
	
	auto A = Matrix(3, 2);
	scope(exit) A.freeMem();
	
	auto B = Matrix(2, 3);
	scope(exit) B.freeMem();
	
	auto C = Matrix(3, 3);
	scope(exit) C.freeMem();
	
	A.each!"a = i";
	B.each!"a = i";
	
	gemm(A, false, B, false, C, cublasHandle);
	cudaDeviceSynchronize();
	
	// cuBLAS is column-major
	float[] result = [
		 3,  4,  5,
		 9, 14, 19,
		15, 24, 33
	];
	
	assert (equal!approxEqual(C.values, result));
	
	C.freeMem();
	C = Matrix(2, 2);
	
	gemm(A, true, B, true, C, cublasHandle);
	cudaDeviceSynchronize();
	
	result = [
		10, 28,
		13, 40
	];
	
	assert (equal!approxEqual(C.values, result));
}

/**
 * A convenient wrapper around cublasSgeam.
 *
 * Performs the matrix-matrix addition C = αop(A) + βop(B).
 *
 * Despite this function is similar to original cublasSgeam function it provides a better interface.
 * Instead of taking raw pointers data and cryptic dimentions parameters it takes matrices and
 * figures out dimentions on its own. Also this function provides better control over dimentions
 * of the matrises.
 *
 * Params:
 *     alpha = α constant multiplier.
 *     A = The first matrix.
 *     transA = Whether to transpose matrix A or not.
 *     beta = β constant multiplier.
 *     B = The second matrix.
 *     transB = Whether to transpose matrix B or not.
 *     C = Output matrix.
 *     cublasHandle = Cublas handle.
 */
void geam(in float alpha, in Matrix A, in bool transA, in float beta, in Matrix B, in bool transB, ref Matrix C, cublasHandle_t cublasHandle) nothrow
{
	cublasOperation_t opA = transA ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
	cublasOperation_t opB = transB ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
	
	int m = transA ? A.cols : A.rows;
	int n = transB ? B.rows : B.cols;
	
	int lda = A.rows;
	int ldb = B.rows;
	int ldc = C.rows;
	
	if (C.rows != m || C.cols != n)
		throw new Error("Illegal matrix-matrix addition.");
	
	cublasSgeam(
		cublasHandle,
		opA, opB,
		m, n,
		&alpha,
		A.ptr, lda,
		&beta,
		B.ptr, ldb,
		C.ptr, ldc
	);
}

///
unittest
{
	mixin(writeTest!geam);
	
	auto A = Matrix(3, 2);
	scope(exit) A.freeMem();
	
	auto B = Matrix(3, 2);
	scope(exit) B.freeMem();
	
	auto C = Matrix(3, 2);
	scope(exit) C.freeMem();
	
	A.each!"a = i";
	B.each!"a = i";
	
	geam(1, A, false, 2, B, false, C, cublasHandle);
	
	float[] result = [
		0,  3,  6,
		9, 12, 15
	];
	
	cudaDeviceSynchronize();
	
	assert (equal!approxEqual(C.values, result));
	
	C.freeMem();
	C = Matrix(2, 3);
	
	geam(1, A, true, 2, B, true, C, cublasHandle);
	
	result = [
		0,  9,
		3, 12,
		6, 15
	];
	
	cudaDeviceSynchronize();
	
	assert (equal!approxEqual(C.values, result));
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
void transpose(in Matrix A, ref Matrix C, cublasHandle_t cublasHandle) nothrow
{
	if (A.rows != C.cols || A.cols != C.rows)
		throw new Error(
			"Illegal matrix transpose from %dx%d to %dx%d"
			.format(A.rows, A.cols, C.rows, C.cols)
		);
	
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

