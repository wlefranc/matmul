#include <iostream>
#include <matrix.h>
#include <time.h>
#include <assert.h>

int main()
{
    // For the sake of simplicity, please choose sizes that are multiple of 256
    unsigned long N = 4096;
    Matrix A(N,N,1);
    Matrix B(N,N,1);

    clock_t tStart=clock();
    Matrix C = Matrix::basic_mult(A,B);
    assert(C == Matrix(N, N, N));
    printf("Time taken for basic_mult: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    Matrix D = Matrix::block_mult_inplace(A,B);
    assert(D == Matrix(N, N, N));
    printf("Time taken for block_mult_inplace: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    Matrix E = Matrix::block_mult_copy(A,B);
    assert(E == Matrix(N, N, N));
    printf("Time taken for block_mult_copy: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    Matrix F = Matrix::block_mult_copy_sse(A,B);
    assert(F == Matrix(N, N, N));
    printf("Time taken for block_mult_copy_sse: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}
