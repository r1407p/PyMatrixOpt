import pytest
import timeit
import Matrix
import random

class Test_Matrix_Performance():
    
    def test_multiply_cache_optimization_performance(self):
        
        for i in range(12):
            size1 = 2**i
            size2 = 2**i
            size3 = 2**i    
            oriMatrix1 = Matrix.Matrix(size1, size2)
            oriMatrix2 = Matrix.Matrix(size2, size3)
            for i in range(size1):
                for j in range(size2):
                    oriMatrix1[i, j] = random.randint(1, 1000)
            for i in range(size2):
                for j in range(size3):
                    oriMatrix2[i, j] = random.randint(1, 1000)
            
            ns = dict(Matrix=Matrix, oriMatrix1=oriMatrix1, oriMatrix2=oriMatrix2)
            t_naive = timeit.Timer('Matrix.matrix_multiply_naive(oriMatrix1, oriMatrix2)', globals=ns)
            t_strassen = timeit.Timer('Matrix.matrix_multiply_strassen(oriMatrix1, oriMatrix2)', globals=ns)
            t_coppersmith_winograd = timeit.Timer('Matrix.matrix_multiply_coppersmith_winograd(oriMatrix1, oriMatrix2)', globals=ns)
            
            with open("performance.txt", "a") as f:
                f.write("Size: " + str(size1) + "\n")
                f.write("Naive: " + str(t_naive.timeit(number=1)) + "\n")
                f.write("Strassen: " + str(t_strassen.timeit(number=1)) + "\n")
                f.write("Coppersmith-Winograd: " + str(t_coppersmith_winograd.timeit(number=1)) + "\n")
                f.write("\n")
        