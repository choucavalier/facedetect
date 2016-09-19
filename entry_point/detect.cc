#include <iostream>

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

  //for(const auto& box : bounding_boxes)
  //{
    //std::cout << "face at (x = " << box.x << ", y = " << box.y << ") "
              //<< "width: " << box.w << ", height = " << box.h << std::endl;
  //}
}
