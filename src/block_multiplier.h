#include <thread>
#include <atomic>
#include "threadsafe_stack.h"
#include "matrix.h"

class BlockMultiplier
{
private:
  std::vector<std::thread> m_threads;
  unsigned int m_max_threads;
  std::atomic<bool> m_end_of_data;
  std::vector<std::vector<Matrix>>& m_res;
  const std::vector<std::vector<Matrix>>& m_lhs;
  const std::vector<std::vector<Matrix>>& m_rhs;

  struct block_context
  {
    unsigned long idx_rows_res;
    unsigned long idx_cols_res;
  };

  threadsafe_stack<block_context> m_stack;

public:
  BlockMultiplier(std::vector<std::vector<Matrix>>& res,
                  const std::vector<std::vector<Matrix>>& lhs,
                  const std::vector<std::vector<Matrix>>& rhs);
  ~BlockMultiplier();
  
  void add_block(const unsigned long idx_rows_res,
                 const unsigned long idx_cols_res);
  void try_multiply();
  void multiply(const block_context blk_ctx);
  void multiply_thread();
  unsigned int get_max_threads() const { return m_max_threads; }
  size_t get_active_threads() const { return m_threads.size(); }
  void create_threads(); 
  void join();
};
