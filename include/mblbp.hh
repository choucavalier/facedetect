#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "window.hh"

struct mblbp_feature
{
  mblbp_feature(int x, int y, int block_width, int block_height); // constructor

  bool operator==(const mblbp_feature &other);

  int x, y; // horizontal (x, y) offset relative to the origin of the window
  int block_w; // block width, must be a multiple of 3
  int block_h; // block height, must be a multiple of 3
};

using mblbp_features = std::vector<mblbp_feature>;

/* Calculate a MB-LBP feature at a specific window
** The feature will be scaled accordingly with window.scale
**
** Parameters
** ----------
** integral : cv::Mat
**     Integral representation of the image
**
** potential_window : window
**     Window that needs to be tested for a face
**
** feature : mblbp_feature
**     Feature to calculate
**
** Return
** ------
** feature_value : int
**     The calculated value of the feature
*/
int mblbp_calculate_feature(const cv::Mat &integral,
                            const window &potential_window,
                            const mblbp_feature &feature);

/* Generate all MB-LBP features inside a window
** The size of the window is defined in "parameters.hh"
**
** This function is mainly used for calculating all the MB-LBP features when
** training the classifier. The classifier is the result of a selection of the
** best MB-LBP features. To select the best MB-LBP features, they need to all be
** calculated, and that is the purpose of this function.
**
** Return
** ------
** features : std::vector<mblbp_feature>
**     All features contained in the window
*/
std::vector<mblbp_feature> mblbp_all_features();
