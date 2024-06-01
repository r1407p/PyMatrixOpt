import pytest
import timeit
import Matrix
import random

class Test_Matrix_Performance():
    
    def test_multiply_performance_block_for_tile(self):
        i = 10
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
        naive_cache_optimized = timeit.Timer(f'Matrix.matrix_multiply_naive_cache_optimized(mat1, mat2)', setup=setup)
        with open("performance_tile_block_size.txt", "a") as f:
            f.write("Size: " + str(size) + "\n")
            f.write("Naive: " + str(min(naive.repeat(repeat=repeat, number=1))) + "\n")
            f.write("Cache-optimized: " + str(min(naive_cache_optimized.repeat(repeat=repeat, number=1))) + "\n")
            f.write("\n")
    
        for block_size in range(4, 32+4, 4):
            tile = timeit.Timer(f'Matrix.matrix_multiply_naive_tile(mat1, mat2, {block_size})', setup=setup)
            cache_optimized_tile = timeit.Timer(f'Matrix.matrix_multiply_naive_cache_optimized_tile(mat1, mat2, {block_size})', setup=setup)  
            with open("performance_tile_block_size.txt", "a") as f:
                f.write("Block size: " + str(block_size) + "\n")
                f.write("Tile: " + str(min(tile.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Cache-optimized-tile: " + str(min(cache_optimized_tile.repeat(repeat=repeat, number=1))) + "\n")
                f.write("\n")
                
    def test_mutiply_performance_cache_optimization_size(self):
        
        for i in range(12):
            size = 2**i
            repeat = 1
            best_tile_block_size = 8
            best_cache_optimized_tile_block_size = 4
            
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
            naive_cache_optimized = timeit.Timer(f'Matrix.matrix_multiply_naive_cache_optimized(mat1, mat2)', setup=setup)
            tile = timeit.Timer(f'Matrix.matrix_multiply_naive_tile(mat1, mat2, {best_tile_block_size})', setup=setup)
            cache_optimized_tile = timeit.Timer(f'Matrix.matrix_multiply_naive_cache_optimized_tile(mat1, mat2, {best_cache_optimized_tile_block_size})', setup=setup)
            with open("performance_cache_optimization_size.txt", "a") as f:
                f.write("Size: " + str(size) + "\n")
                f.write("Naive: " + str(min(naive.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Cache-optimized: " + str(min(naive_cache_optimized.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Tile: " + str(min(tile.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Cache-optimized-tile: " + str(min(cache_optimized_tile.repeat(repeat=repeat, number=1))) + "\n")
                f.write("\n")
                
    def test_multiply_thread(self):
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
            cache_optimized_tile = timeit.Timer(f'Matrix.matrix_multiply_naive_cache_optimized_tile(mat1, mat2, 4)', setup=setup)
            multi_thread = timeit.Timer(f'Matrix.matrix_multiply_naive_cache_optimized_tile_thread(mat1, mat2, 4)', setup=setup)
            with open("performance_thread.txt", "a") as f:
                f.write("Size: " + str(size) + "\n")
                f.write("Cache-optimized-tile: " + str(min(cache_optimized_tile.repeat(repeat=repeat, number=1))) + "\n")
                f.write("Multi-thread: " + str(min(multi_thread.repeat(repeat=repeat, number=1))) + "\n")
                f.write("\n")
                