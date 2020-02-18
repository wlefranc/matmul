#include "matrix.h"
#include <functional>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h> // for sse instructions

Matrix::Matrix(const Matrix& m) : N(m.N), M(m.M), t(new int[N*M])
{
    for(size_t i = 0; i < m.N; ++i)
    {
        for(size_t j = 0; j < m.M; ++j)
        {
            (*this)(i,j) = m(i,j);
        }
    }
}

Matrix::Matrix(Matrix&& m) : N(m.N), M(m.M), t(m.t)
{
    m.t = nullptr;
}

bool Matrix::operator==(const Matrix& m)
{
	if (N != m.N || M != m.M)
		return false;

	return memcmp(t, m.t, sizeof(int)*N*M) == 0;

}

Matrix Matrix::block_mult_copy_sse(const Matrix& A, const Matrix& B)
{
    const unsigned long N = A.get_rows();
    const unsigned long M = B.get_cols();
    const unsigned long K = A.get_cols();

    Matrix C(N,M);

    const unsigned long SM=256uL;
    const unsigned long NN=N/SM;
    const unsigned long MM=M/SM;
    const unsigned long KK=K/SM;

    std::vector<std::vector< Matrix >> AA(NN, std::vector<Matrix>(KK, {SM,SM}));
    std::vector<std::vector< Matrix >> BB(KK, std::vector<Matrix>(MM, {SM,SM}));
    std::vector<std::vector< Matrix >> CC(NN, std::vector<Matrix>(MM, {SM,SM}));

    Matrix::from_matrix_to_ppmatrix(A,AA,SM);
    Matrix::from_matrix_to_ppmatrix(B,BB,SM);

    unsigned long i, k;
    int *rres, *rmul1, *rmul2;

    for(size_t ii = 0; ii < NN; ++ii)
    {
        for(size_t jj = 0; jj < MM; ++jj)
	{
            for(size_t kk = 0; kk < KK; ++kk)
	    {
                for(i = 0, rres = CC[ii][jj].t, rmul1 = AA[ii][kk].t;
                    i < SM;
                    ++i, rres += SM, rmul1 += SM)
		{
                    for(k = 0, rmul2 = BB[kk][jj].t;
                        k < SM;
                        ++k, rmul2 += SM)
		    {
			size_t j = 0;
			__m128i* simd_rmul2 = reinterpret_cast<__m128i*>(rmul2); 

			for(__m128i* simd_res = reinterpret_cast<__m128i*>(rres);
			    j < SM;
			    ++simd_res, ++simd_rmul2, j += 4)
			{
				auto mult = _mm_mullo_epi32(_mm_set1_epi32(rmul1[k]), *simd_rmul2);
				//auto add = _mm_add_epi32(*simd_res, mult); 
				*simd_res = _mm_add_epi32(*simd_res, mult); 

				//_mm_store_si128(simd_res, add);
			}
		    }
		}
	    }
	}
    }

    Matrix::from_ppmatrix_to_matrix(CC,C,SM);

    return C;
}

Matrix Matrix::block_mult_inplace(const Matrix& A, const Matrix& B)
{
    const unsigned long N = A.get_rows();
    const unsigned long M = B.get_cols();
    const unsigned long K = A.get_cols();

    Matrix C(N,M);

    const unsigned long SM = 256uL;

    unsigned long i,k;
    int *rres,*rmul1,*rmul2;

    for(size_t ii = 0; ii < N; ii += SM)
    {
        for(size_t jj = 0; jj < M; jj += SM)
	{
            for(size_t kk = 0; kk < K; kk += SM)
	    {
                for(i = 0, rres = &C.t[ii*M+jj], rmul1 = &A.t[ii*K+kk];
                    i < SM;
                    ++i, rres += M, rmul1 += K)
		{
                    for(k = 0, rmul2 = &B.t[kk*M+jj];
                        k < SM;
                        ++k, rmul2 += N)
		    {
                        for(size_t j = 0; j < SM; ++j)
                            rres[j] += rmul1[k] * rmul2[j];
		    }
		}
	    }
	}
    }

    return C;
}

void Matrix::from_matrix_to_ppmatrix(const Matrix& M,
                                     std::vector<std::vector<Matrix>>& MM,
                                     unsigned long SM)
{
    const unsigned long N = M.get_rows();
    const unsigned long K = M.get_cols();

    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < K; ++j)
        {
            auto ii = i/SM, jj = j/SM;
            auto i2 = i%SM, j2 = j%SM;
            Matrix& m = MM[ii][jj];
	    m(i2,j2) = M(i,j);
        }
    }
}

void Matrix::from_ppmatrix_to_matrix(const std::vector<std::vector<Matrix>>& MM,
                                     Matrix& M,
                                     unsigned long SM)
{
    const unsigned long N = M.get_rows();
    const unsigned long K = M.get_cols();

    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < K; ++j)
        {
            auto ii = i/SM, jj = j/SM;
            auto i2 = i%SM, j2= j%SM;
            const Matrix& m = MM[ii][jj];
	    M(i,j) = m(i2,j2);
        }
    }
}

Matrix Matrix::block_mult_copy(const Matrix& A,const Matrix& B)
{
    const unsigned long N = A.get_rows();
    const unsigned long M = B.get_cols();
    const unsigned long K = A.get_cols();

    Matrix C(N,M);

    const unsigned long SM=256uL;
    const unsigned long NN=N/SM;
    const unsigned long MM=M/SM;
    const unsigned long KK=K/SM;

    std::vector<std::vector< Matrix >> AA(NN, std::vector<Matrix>(KK, {SM,SM}));
    std::vector<std::vector< Matrix >> BB(KK, std::vector<Matrix>(MM, {SM,SM}));
    std::vector<std::vector< Matrix >> CC(NN, std::vector<Matrix>(MM, {SM,SM}));

    Matrix::from_matrix_to_ppmatrix(A,AA,SM);
    Matrix::from_matrix_to_ppmatrix(B,BB,SM);

    unsigned long i,k;
    int *rres,*rmul1,*rmul2;

    for(size_t ii = 0; ii < NN; ++ii)
    {
        for(size_t jj = 0; jj < MM; ++jj)
	{
            for(size_t kk = 0; kk < KK; ++kk)
	    {
                for(i = 0, rres = CC[ii][jj].t, rmul1 = AA[ii][kk].t;
                    i < SM;
                    ++i, rres += SM, rmul1 += SM)
		{
                    for(k = 0, rmul2 = BB[kk][jj].t;
                        k < SM;
                        ++k, rmul2 += SM)
		    {
                        for(size_t j = 0; j < SM; ++j)
                            rres[j] += rmul1[k] * rmul2[j];
		    }
		}
	    }
	}
    }

    Matrix::from_ppmatrix_to_matrix(CC,C,SM);

    return C;
}

Matrix Matrix::basic_mult(const Matrix& A,const Matrix& B)
{
    const unsigned long N = A.get_rows();
    const unsigned long M = B.get_cols();
    const unsigned long K = A.get_cols();

    Matrix C(N,M);

    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j < M; ++j)
            for(size_t k = 0; k < K; ++k)
                C(i,j) += A(i,k) * B(k,j);

    return C;
}

void Matrix::dump(std::ostream& os) const
{
	for(size_t i = 0; i < N; ++i)
	{
		os << "[ ";
		for(size_t j = 0; j < M; ++j)
		{
			os << (*this)(i,j) << " ";
		}
		os << "]\n";
	}
}

















