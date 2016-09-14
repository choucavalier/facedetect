#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "window.hh"

struct mblbp_feature
{
  mblbp_feature(int x, int y, int block_width, int block_height);
  int x; // horizontal offset
  int y; // vertical offset
  int block_width; // must be a multiple of 3
  int block_height; // must be a multiple of 3
};

using mblbp_features = std::vector<mblbp_feature>;

int mblbp_calculate_feature(const cv::Mat &integral,
                            const window &potential_window,
                            const mblbp_feature &feature);
