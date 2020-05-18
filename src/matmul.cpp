#include <iostream>
#include <matrix.h>
#include <assert.h>
#include <chrono>

int main()
{
    unsigned long N = 4094;
    Matrix A(N,N,1);
    Matrix B(N,N,1);

    std::cout << "block_mult_parallel..."; std::cout.flush();
    auto t_start = std::chrono::high_resolution_clock::now();
    Matrix G = Matrix::block_mult_parallel(A,B);
    auto t_end = std::chrono::high_resolution_clock::now();
    assert(G == Matrix(N, N, N));
    std::cout << std::chrono::duration<double, std::milli>(t_end-t_start).count() << "ms" << std::endl;

    std::cout << "block_mult_copy_sse..."; std::cout.flush();
    t_start = std::chrono::high_resolution_clock::now();
    Matrix F = Matrix::block_mult_copy_sse(A,B);
    t_end = std::chrono::high_resolution_clock::now();
    assert(F == Matrix(N, N, N));
    std::cout << std::chrono::duration<double, std::milli>(t_end-t_start).count() << "ms" << std::endl;

    std::cout << "block_mult_copy..."; std::cout.flush();
    t_start = std::chrono::high_resolution_clock::now();
    Matrix E = Matrix::block_mult_copy(A,B);
    t_end = std::chrono::high_resolution_clock::now();
    assert(E == Matrix(N, N, N));
    std::cout << std::chrono::duration<double, std::milli>(t_end-t_start).count() << "ms" << std::endl;

    std::cout << "block_mult_inplace..."; std::cout.flush();
    t_start = std::chrono::high_resolution_clock::now();
    Matrix D = Matrix::block_mult_inplace(A,B);
    t_end = std::chrono::high_resolution_clock::now();
    assert(D == Matrix(N, N, N));
    std::cout << std::chrono::duration<double, std::milli>(t_end-t_start).count() << "ms" << std::endl;

    /*std::cout << "basic_mult..."; std::cout.flush();
    tStart=clock();
    Matrix C = Matrix::basic_mult(A,B);
    assert(C == Matrix(N, N, N));
    printf("%.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);*/

    return 0;
}
