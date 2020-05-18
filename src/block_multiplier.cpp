#include <cassert>
#include "block_multiplier.h"

BlockMultiplier::BlockMultiplier(std::vector<std::vector<Matrix>>& res,
                                 const std::vector<std::vector<Matrix>>& lhs,
                                 const std::vector<std::vector<Matrix>>& rhs) :
  m_threads(),
  m_max_threads(std::thread::hardware_concurrency()-1),
  m_end_of_data(false),
  m_res(res),
  m_lhs(lhs),
  m_rhs(rhs),
  m_stack()  
{
}

BlockMultiplier::~BlockMultiplier()
{
}

void BlockMultiplier::join()
{
  while (!m_stack.empty())
    try_multiply();

  m_end_of_data = true;
  for(auto& t : m_threads)
    t.join();
}
   
void BlockMultiplier::add_block(const unsigned long idx_rows_res,
                                const unsigned long idx_cols_res)
{
  m_stack.push({idx_rows_res, idx_cols_res});
}

void BlockMultiplier::multiply_thread()
{
  while (!m_end_of_data)
  {
    try_multiply();
    std::this_thread::yield();
  }
}

void BlockMultiplier::try_multiply()
{
  const std::shared_ptr<block_context> blk_ctx = m_stack.pop();
  if (blk_ctx)
    multiply(*blk_ctx);
}

void BlockMultiplier::multiply(const BlockMultiplier::block_context blk_ctx)
{
  Matrix& res_block = m_res[blk_ctx.idx_rows_res][blk_ctx.idx_cols_res];

  for(unsigned long kk = 0; kk < m_rhs.size(); ++kk)
  {
    const Matrix& lhs_block = m_lhs[blk_ctx.idx_rows_res][kk];
    const Matrix& rhs_block = m_rhs[kk][blk_ctx.idx_cols_res];

    for(unsigned long i = 0; i < res_block.get_rows(); ++i)
    {
      for(unsigned long k = 0; k < lhs_block.get_cols(); ++k)
      {
        for(unsigned long j = 0; j < res_block.get_cols(); ++j)
	{
	  res_block(i,j) += lhs_block(i,k) * rhs_block(k,j);
	}
      }
    }
  }
}

void BlockMultiplier::create_threads()
{
    for(unsigned int i = 0; i < m_max_threads; ++i)
      m_threads.push_back(std::thread(&multiply_thread, this));
}
