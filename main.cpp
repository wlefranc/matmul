#include <iostream>
#include <matrix.h>
#include <time.h>
#include <assert.h>

int main()
{
    // For the sake of simplicity, please choose sizes that are multiple of 256
    unsigned long N = 4096;
    Matrix<int> A(N,N,1);
    Matrix<int> B(N,N,1);

    clock_t tStart=clock();
    /*Matrix<int> C = Matrix<int>::basic_mult(A,B);
    assert(C == Matrix<int>(N, N, N));
    printf("Time taken for basic_mult: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);*/

    tStart = clock();
    Matrix<int> D = Matrix<int>::block_mult_inplace(A,B);
    assert(D == Matrix<int>(N, N, N));
    printf("Time taken for block_mult_inplace: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    Matrix<int> E = Matrix<int>::block_mult_copy(A,B);
    assert(E == Matrix<int>(N, N, N));
    printf("Time taken for block_mult_copy: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    Matrix<int> F = Matrix<int>::block_mult_copy_sse(A,B);
    //std::cout << F << std::endl;
    assert(F == Matrix<int>(N, N, N));
    printf("Time taken for block_mult_copy_sse: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}
