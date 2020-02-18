#include <iostream>
#include <matrix.h>
#include <time.h>
#include <assert.h>

int main()
{
    unsigned long N = 4096;
    Matrix A(N,N,1);
    Matrix B(N,N,1);

    std::cout << "block_mult_copy_sse..."; std::cout.flush();
    clock_t tStart=clock();
    Matrix F = Matrix::block_mult_copy_sse(A,B);
    assert(F == Matrix(N, N, N));
    printf("%.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    std::cout << "block_mult_copy..."; std::cout.flush();
    tStart = clock();
    Matrix E = Matrix::block_mult_copy(A,B);
    assert(E == Matrix(N, N, N));
    printf("%.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    std::cout << "block_mult_inplace..."; std::cout.flush();
    tStart = clock();
    Matrix D = Matrix::block_mult_inplace(A,B);
    assert(D == Matrix(N, N, N));
    printf("%.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    std::cout << "basic_mult..."; std::cout.flush();
    Matrix C = Matrix::basic_mult(A,B);
    assert(C == Matrix(N, N, N));
    printf("%.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}
