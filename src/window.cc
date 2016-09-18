#include "params.hh"
#include "window.hh"

std::vector<window> get_potential_windows(const int img_w, const int img_h)
{
  std::vector<window> potential_windows;

  double w = initial_window_w; // current window width
  double h = initial_window_h; // current window height
  double x = 0.0; // current horizontal offset
  double y = 0.0; // current vertical offset
  double scale = 1.0;

  while(w <= max_window_w && h <= max_window_h)
  {
    for(double x = 0.0; x < img_w - w; x += scale * shift_delta)
      for(double y = 0.0; y < img_h - h; y += scale * shift_delta)
        potential_windows.push_back(window(x, y, w, h, scale));

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
  // TODO
  return {};
}
