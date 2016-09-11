#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "detect.hh"

std::vector<bbox> detect(std::string img_path)
{
  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat integral_img;
  cv::integral(img, integral_img);
}
