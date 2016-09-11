#pragma once

#include <string>
#include <vector>

struct bbox
{
  int x;
  int y;
  int width;
  int height;
};

/*
** Detect bounding boxes around faces in the given image
*/
std::vector<bbox> detect(std::string img_path);
