#include <iostream>

#include "detect.hh"

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cout << "usage: ./detect 'path/to/image'" << std::endl;
    return 1;
  }

  std::cout << "detecting faces in " << argv[1] << std::endl;

  std::vector<bbox> bounding_boxes = detect(argv[1]);
}
