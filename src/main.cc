#include <iostream>

int main(int argc, char **argv)
{
  if(argc < 2)
    std::cout << "usage: ./detect 'path/to/image'" << std::endl;

  std::cout << "detecting faces in " << argv[1] << std::endl;
}
