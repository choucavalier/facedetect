#pragma once

#include <vector>

// a window that might contain a face
struct window
{
  // basic constructor
  window(double x, double y, double w, double h, double scale)
    : x(x), y(y), w(w), h(h), scale(scale) {}
  // absolute offset
  double x;
  double y;
  // rectangle dimensions
  double w;
  double h;
  // scaling from the initial window size
  // e.g. if the initial window size is set to 20x24
  //      and the scale is set to 1.5
  double scale;
};

struct bbox
{
  bbox(int x, int y, int w, int h) : x(x), y(y), w(w), h(h) {}
  int x;
  int y;
  int w;
  int h;
};

// get all windows potentially containing a face
std::vector<window> get_potential_windows(int img_w, int img_h);

// aggregates windows into bounding boxes
// img_size is needed to make sure the bounding box isn't out of bound
std::vector<bbox> aggregate_windows(int img_w, int img_h,
                                    const std::vector<window> &windows);