#include "mblbp.hh"

mblbp_feature::mblbp_feature(int x, int y, int block_width, int block_height) :
  x(x), y(y), block_width(block_width), block_height(block_height)
{
  if(block_width % 3 != 0 || block_height % 3 != 0)
    throw std::invalid_argument("wrong mblbp block size");
}

int mblbp_calculate_feature(const cv::Mat &integral,
                            const window &potential_window,
                            const mblbp_feature &feature)
{
  return 0;
}
