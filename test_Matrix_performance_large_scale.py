import pytest
import timeit
import Matrix
import random

class Test_Matrix_Performance():
    
    def test_multiply_performance(self):
        for i in range(12):
            size = 2**i
            repeat = 1
            
            setup = f'''
import Matrix
size = {size}
mat1 = Matrix.Matrix(size,size)
mat2 = Matrix.Matrix(size,size)
for it in range(size):
    for jt in range(size):
        mat1[it, jt] = it * size + jt + 1
        mat2[it, jt] = it * size + jt + 1
'''
            
            naive = timeit.Timer('Matrix.matrix_multiply_naive(mat1, mat2)', setup=setup)
            strassen = timeit.Timer('Matrix.matrix_multiply_strassen(mat1, mat2)', setup=setup)
            coppersmith_winograd = timeit.Timer('Matrix.matrix_multiply_coppersmith_winograd(mat1, mat2)', setup=setup)

            with open("performance.txt", "a") as f:
                f.write("Size: " + str(size) + "\n")
                f.write("Naive: " + str(min(naive.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Strassen: " + str(min(strassen.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Coppersmith-Winograd: " + str(min(coppersmith_winograd.repeat(repeat=repeat, number=1))) + "\n")
                f.write("\n")
