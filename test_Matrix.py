import pytest
import timeit
import Matrix
import random

class Test_Matrix():
    def test_constructer(self):
    
        col = random.randint(1, 1000)
        row = random.randint(1, 1000)
        matrix = Matrix.Matrix(row, col)
        
        assert matrix.nrow == row
        assert matrix.ncol == col
        for i in range(row):
            for j in range(col):
                assert matrix[i, j] == 0 
                
    def test_multiply_correctness(self):
        size1 = random.randint(1, 1000)
        size2 = random.randint(1, 1000)
        size3 = random.randint(1, 1000)
        oriMatrix1 = Matrix.Matrix(size1, size2)
        oriMatrix2 = Matrix.Matrix(size2, size3)
        for i in range(size1):
            for j in range(size2):
                oriMatrix1[i, j] = random.randint(1, 1000)
        for i in range(size2):
            for j in range(size3):
                oriMatrix2[i, j] = random.randint(1, 1000)
        retMatrix1 = Matrix.matrix_multiply_naive(oriMatrix1, oriMatrix2)
        retMatrix2 = Matrix.matrix_multiply_strassen(oriMatrix1, oriMatrix2)
        retMatrix3 = Matrix.matrix_multiply_coppersmith_winograd(oriMatrix1, oriMatrix2)
        
        for i in range(size1):
            for j in range(size3):
                assert retMatrix1[i, j] == retMatrix2[i, j] == retMatrix3[i, j]
                
    def make_matrices(self, size):

        mat1 = Matrix.Matrix(size,size)
        mat2 = Matrix.Matrix(size,size)
        mat3 = Matrix.Matrix(size,size)

        for it in range(size):
            for jt in range(size):
                mat1[it, jt] = it * size + jt + 1
                mat2[it, jt] = it * size + jt + 1
                mat3[it, jt] = 0

        return mat1, mat2, mat3

    def test_basic_create(self):
        size = 100
        mat1, _, _ = self.make_matrices(size)
        for i in range(mat1.nrow):
            for j in range(mat1.ncol):
                assert i * size + j + 1 == mat1[i,j]

    def test_basic(self):

        size = 100
        mat1, mat2, mat3 = self.make_matrices(size)

        assert size == mat1.nrow
        assert size == mat1.ncol
        assert size == mat2.nrow
        assert size == mat2.ncol
        assert size == mat3.nrow
        assert size == mat3.ncol

        assert 2 == mat1[0,1]
        assert size+2 == mat1[1,1]
        assert size*2 == mat1[1,size-1]
        assert size*size == mat1[size-1,size-1]

        for i in range(mat1.nrow):
            for j in range(mat1.ncol):
                assert 0 != mat1[i,j]
                assert mat1[i,j] == mat2[i,j]
                assert 0 == mat3[i,j]

        assert mat1 == mat2
        assert mat1 is not mat2

    def test_match_naive_strassen(self):
        size = 100
        mat1, mat2, _ = self.make_matrices(size)
        ret_naive = Matrix.matrix_multiply_naive(mat1, mat2)
        ret_strassen = Matrix.matrix_multiply_strassen(mat1, mat2)
        assert size == ret_naive.nrow
        assert size == ret_naive.ncol
        assert size == ret_strassen.nrow
        assert size == ret_strassen.ncol

        for i in range(ret_naive.nrow):
            for j in range(ret_naive.ncol):
                assert mat1[i,j] is not ret_strassen[i,j]
                assert ret_naive[i,j] == ret_strassen[i,j]

    def test_match_naive_coppersmith_winograd(self):

        size = 100
        mat1, mat2, _ = self.make_matrices(size)

        ret_naive = Matrix.matrix_multiply_naive(mat1, mat2)
        ret_coppersmith_winograd = Matrix.matrix_multiply_coppersmith_winograd(mat1, mat2)

        assert size == ret_naive.nrow
        assert size == ret_naive.ncol
        assert size == ret_coppersmith_winograd.nrow
        assert size == ret_coppersmith_winograd.ncol

        for i in range(ret_naive.nrow):
            for j in range(ret_naive.ncol):
                assert mat1[i,j] is not ret_coppersmith_winograd[i,j]
                assert ret_naive[i,j] == ret_coppersmith_winograd[i,j]

    def test_zero(self):
        size = 100
        mat1, mat2, mat3 = self.make_matrices(size)

        ret_naive = Matrix.matrix_multiply_naive(mat1, mat3)
        ret_strassen = Matrix.matrix_multiply_strassen(mat1, mat3)
        ret_coppersmith_winograd = Matrix.matrix_multiply_coppersmith_winograd(mat1, mat3)

        assert size == ret_naive.nrow
        assert size == ret_naive.ncol
        assert size == ret_strassen.nrow
        assert size == ret_strassen.ncol
        assert size == ret_coppersmith_winograd.nrow
        assert size == ret_coppersmith_winograd.ncol
        
        for i in range(ret_naive.nrow):
            for j in range(ret_naive.ncol):
                assert 0 == ret_naive[i, j]
                assert 0 == ret_strassen[i, j]
                assert 0 == ret_coppersmith_winograd[i, j]
