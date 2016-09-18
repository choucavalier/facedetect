#include <iostream>

#include "detect.hh"
#include "mblbp.hh"
#include "window.hh"

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cout << "usage: ./detect 'path/to/image'" << std::endl;
    return 1;
  }

  auto potential_windows = get_potential_windows(320, 200);

  std::cout << potential_windows.size() << " potential windows" << std::endl;
  return 0;

  std::string img_path(argv[1]);
  std::string classifier_path("bullshit_classifier_path");

  std::cout << "detecting faces in " << argv[1] << std::endl;

  std::vector<bbox> bounding_boxes = detect(img_path, classifier_path);

  for(const auto& box : bounding_boxes)
  {
    std::cout << "face at (x = " << box.x << ", y = " << box.y << ") "
              << "width: " << box.w << ", height = " << box.h << std::endl;
  }
}
