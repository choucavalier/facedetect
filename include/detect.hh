#pragma once

#include <string>
#include <vector>

struct bbox
{
  bbox(int x, int y, int w, int h) : x(x), y(y), w(w), h(h)
  {
  }

  int x;
  int y;
  int w;
  int h;
};

/*
** Detect bounding boxes around faces in the given image
*/
std::vector<bbox> detect(std::string img_path);
