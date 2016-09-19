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

static double avg_block_value(const cv::Mat &integral,
                              const int x1, const int y1,
                              const int x2, const int y2)
{
  uchar a = integral.at<uchar>(x1, y1);
  uchar b = integral.at<uchar>(x2, y1);
  uchar c = integral.at<uchar>(x1, y2);
  uchar d = integral.at<uchar>(x2, y2);
  int sum = d - b - c + a;

  return (double)sum * (y2 - y1) / (x2 - x1);
}

unsigned char mblbp_calculate_feature(const cv::Mat &integral,
                                      const window &window,
                                      const mblbp_feature &feature)
{
  double avg_block_values[9];
  int x1, y1, x2, y2;
  double base_x = window.x + (feature.x * window.scale);
  double base_y = window.y + (feature.y * window.scale);
  double block_w = feature.block_w * window.scale / 3;
  double block_h = feature.block_h * window.scale / 3;
  for(int i = 0; i < 9; ++i)
  {
    x1 = base_x + (i % 3) * block_w;
    y1 = base_y + (i / 3) * block_h;
    x2 = x1 + block_w;
    y2 = y1 + block_h;
    avg_block_values[i] = avg_block_value(integral, x1, y1, x2, y2);
  }

  int lbp_code = 0;
  double mid_value = avg_block_values[4];
  if(avg_block_values[0] > mid_value) lbp_code |= 1;
  if(avg_block_values[1] > mid_value) lbp_code |= 2;
  if(avg_block_values[2] > mid_value) lbp_code |= 4;
  if(avg_block_values[3] > mid_value) lbp_code |= 128;
  if(avg_block_values[5] > mid_value) lbp_code |= 8;
  if(avg_block_values[6] > mid_value) lbp_code |= 64;
  if(avg_block_values[7] > mid_value) lbp_code |= 32;
  if(avg_block_values[8] > mid_value) lbp_code |= 16;

  return lbp_code;
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
