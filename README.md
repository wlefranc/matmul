# matmul
Fast multiplication of matrices

The purpose of this project is to compare several matrix multiplication algorithms and show how exploiting cache locality
can significantly improve execution speed.
Until now, four versions have been implemented : 
- the basic one, called basic_mult, which completely compute c[i][j] by looping over k : c[i][j] += a[i][k] * b[k][j].
Not very cache-friendly.
- block_mult_inplace, which decomposes the matrix into subblocks and performs operations on each subblock before moving to the next one.
- block_mult_copy, which does the same as above but also rearranges the data of one subblock so that all lines of one subblock 
are contiguous in memory.
- block_mult_copy_sse, same principle as above but with sse intrinsics.

Try modifying the optimization level by changing the OPT_LEVEL variable in the Makefile to see how it affects performance. 
There should not be much difference between block_mult_copy and block_mult_copy_sse with -O3 activated as gcc automatically uses
vectorization at this level of optimization. But there is a huge difference when using -02.

Here are the results with -O2 on two 4096*4096 matrices (with an Intel i3-3227U, 1.90GHz CPU)
- Time taken for block_mult_inplace: 148.28s
- Time taken for block_mult_copy: 116.52s
- Time taken for block_mult_copy_sse: 37.31s

With -O3:
- Time taken for block_mult_inplace: 62.92s
- Time taken for block_mult_copy: 32.31s
- Time taken for block_mult_copy_sse: 30.05s
