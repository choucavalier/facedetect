#include "params.hh"
#include "mblbp.hh"

mblbp_feature::mblbp_feature(int x, int y, int block_w, int block_h) :
  x(x), y(y), block_w(block_w), block_h(block_h)
{
  if(block_w < 3 || block_h < 3)
    throw std::invalid_argument("wrong mblbp block size (req greater than 2)");
  if(block_w % 3 != 0 || block_h % 3 != 0)
    throw std::invalid_argument("wrong mblbp block size (req divisible by 3)");
}

int mblbp_calculate_feature(const cv::Mat &integral,
                            const window &potential_window,
                            const mblbp_feature &feature)
{
  // TODO
  return 0;
}

std::vector<mblbp_feature> mblbp_all_features()
{
  std::vector<mblbp_feature> features;
  for(int block_w = min_block_size; block_w <= max_block_size; block_w += 3)
    for(int block_h = min_block_size; block_h <= max_block_size; block_h += 3)
      for(int x = 0; x <= initial_window_w - block_w; ++x)
        for(int y = 0; y <= initial_window_h - block_h; ++y)
          features.push_back(mblbp_feature(x, y, block_w, block_h));
  return features;
}

bool mblbp_feature::operator==(const mblbp_feature &other)
{
  return this->x == other.x &&
         this->y == other.y &&
         this->block_w == other.block_w &&
         this->block_h == other.block_h;
}
