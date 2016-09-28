#include <iostream>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

int main()
{
  fs::path lfw_path = "data/lfwcrop_grey/faces";
  fs::path positive_path = "data/positive";

  cv::Size window_size(20, 20);

  for(auto& directory_entry : fs::directory_iterator(lfw_path))
  {
    fs::path save_path = positive_path / directory_entry.path().filename();
    save_path.replace_extension(".png");

    cv::Mat img = cv::imread(directory_entry.path().string());
    cv::Mat resized;
    cv::resize(img, resized, window_size);
    cv::imwrite(save_path.string(), resized);
  }
}
