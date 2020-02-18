#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <type_traits>

template<class T>
class Matrix
{
private:
    const unsigned long N;
    const unsigned long M;
    T* t;

public:
    Matrix(unsigned long rows, unsigned long cols, T value = T()) : N(rows), M(cols), t(new T[N*M])
    {
        for(unsigned long i = 0; i < N*M; i++)
            t[i] = value;
    }
    
    Matrix(const Matrix& m);
    Matrix(Matrix&& m);
    ~Matrix()
    {
        delete[] t;
    }

    bool operator==(const Matrix<T>& m);

    T operator()(unsigned long i, unsigned long j) const
    {
	    return t[i*M+j];
    }
    
    T& operator()(unsigned long i, unsigned long j)
    {
	    return t[i*M+j];
    }

    unsigned long get_rows() const { return N; }
    unsigned long get_cols() const { return M; }
    void dump(std::ostream& os) const;

    static Matrix<T> basic_mult(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> block_mult_inplace(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> block_mult_copy(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> block_mult_copy_sse(const Matrix<T>& A, const Matrix<T>& B);

    static void from_matrix_to_ppmatrix(const Matrix<T>& M, std::vector<std::vector<Matrix<T>>>& MM, unsigned long SM);
    static void from_ppmatrix_to_matrix(const std::vector<std::vector<Matrix<T>>>& MM, Matrix<T>& M, unsigned long SM);
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
{
	m.dump(os);
	return os;
}

#include "matrix.hpp"

#endif // MATRIX_H
