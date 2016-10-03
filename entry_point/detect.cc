#include <iostream>

#include <opencv2/opencv.hpp>

#include "detect.hh"

int main(int argc, char **argv)
{
  if(argc < 3)
  {
    std::cout << "usage: ./detect 'path/to/image' 'path/to/classifier'"
              << std::endl;
    return 1;
  }

  std::string img_path(argv[1]);
  std::string classifier_path(argv[2]);

  std::cout << "detecting faces in " << argv[1] << std::endl;

  std::vector<bbox> bounding_boxes = detect(img_path, classifier_path);

  cv::Mat img = cv::imread(img_path);

  for(const auto& box : bounding_boxes)
  {
    /*
    std::cout << "face at (x = " << box.x << ", y = " << box.y << ") "
              << "width: " << box.w << ", height = " << box.h << std::endl;
    */
    cv::rectangle(
        img,
        cv::Point(box.x, box.y),
        cv::Point(box.x + box.w, box.y + box.h),
        cv::Scalar(0, 255, 0)
    );

  }

  cv::imwrite("output.jpg", img);

}
