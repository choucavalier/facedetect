#include "params.hh"
#include "window.hh"

#include <iostream>

std::vector<window> get_potential_windows(const int img_w, const int img_h)
{
  std::cout << "getting potential windows : " << img_w << " , " << img_h << std::endl;
  std::vector<window> potential_windows;

  double w = initial_window_w; // current window width
  double h = initial_window_h; // current window height
  double scale = 1.0; // current scale

  while(w <= max_window_w && h <= max_window_h)
  {
    for(double x = 0.0; x < static_cast<double>(img_w) - w; x += 5.0)
    {
      for(double y = 0.0; y < static_cast<double>(img_h) - h; y += 5.0)
        potential_windows.push_back(window(x, y, w, h, scale));
    }

    // update values for next loop
    scale = scale * scaling_factor;
    w = w * scaling_factor;
    h = h * scaling_factor;
  }

  return potential_windows;
}

std::vector<bbox> aggregate_windows(const int img_w, const int img_h,
                                    const std::vector<window> &windows)
{
  std::vector<bbox> bounding_boxes;

  for(const auto& potential_window : windows)
  {
    int x, y, w, h;
    x = potential_window.x;
    y = potential_window.y;
    w = potential_window.w;
    h = potential_window.h;
    bounding_boxes.push_back(bbox(x, y, w, h));
  }

  // TODO

  return bounding_boxes;
}
