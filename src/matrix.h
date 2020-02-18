#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <type_traits>

class Matrix
{
private:
    const unsigned long N;
    const unsigned long M;
    int* t;

public:
    Matrix(unsigned long rows, unsigned long cols, int value = 0) : N(rows), M(cols), t(new int[N*M])
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

    bool operator==(const Matrix& m);

    int operator()(unsigned long i, unsigned long j) const
    {
	    return t[i*M+j];
    }
    
    int& operator()(unsigned long i, unsigned long j)
    {
	    return t[i*M+j];
    }

    unsigned long get_rows() const { return N; }
    unsigned long get_cols() const { return M; }
    void dump(std::ostream& os) const;

    static Matrix basic_mult(const Matrix& A, const Matrix& B);
    static Matrix block_mult_inplace(const Matrix& A, const Matrix& B);
    static Matrix block_mult_copy(const Matrix& A, const Matrix& B);
    static Matrix block_mult_copy_sse(const Matrix& A, const Matrix& B);

    static void from_matrix_to_ppmatrix(const Matrix& M, std::vector<std::vector<Matrix>>& MM, unsigned long SM);
    static void from_ppmatrix_to_matrix(const std::vector<std::vector<Matrix>>& MM, Matrix& M, unsigned long SM);
};

inline std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
	m.dump(os);
	return os;
}

#endif // MATRIX_H
