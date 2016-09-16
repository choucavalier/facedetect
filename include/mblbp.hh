#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "window.hh"

struct mblbp_feature
{
  // self-explanatory constructor
  mblbp_feature(int x, int y, int block_width, int block_height);
  // horizontal (x, y) offset relative to the origin of the window
  int x, y;
  int block_w; // block width, must be a multiple of 3
  int block_h; // block height, must be a multiple of 3
};

using mblbp_features = std::vector<mblbp_feature>;

int mblbp_calculate_feature(const cv::Mat &integral,
                            const window &potential_window,
                            const mblbp_feature &feature);

std::vector<mblbp_feature> mblbp_all_features();
