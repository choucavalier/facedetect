#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "detect.hh"

// a window that might contain a face
struct window
{
  // basic constructor
  window(double x, double y, double w, double h) : x(x), y(y), w(w), h(h) {}
  // absolute offset
  double x;
  double y;
  // rectangle dimensions
  double w, h;
};

// get all windows potentially containing a face
std::vector<window> get_potential_windows(cv::Size img_size)
{
  return {};
}

// aggregates windows into bounding boxes
// img_size is needed to make sure the bounding box isn't out of bound
static std::vector<bbox> aggregate_windows(cv::Size img_size,
                                           std::vector<window> windows)
{
  return {};
}

std::vector<bbox> detect(std::string img_path)
{
  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Size img_size = img.size();

  std::vector<window> potential_windows = get_potential_windows(img_size);
  std::vector<window> positive_windows;

  return aggregate_windows(img_size, positive_windows);
}
