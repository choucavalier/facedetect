#include "params.hh"
#include "window.hh"

// get all windows potentially containing a face
std::vector<window> get_potential_windows(const int img_w, const int img_h)
{
  double window_width = initial_window_w;
  double window_height = initial_window_h;
  // TODO
  return {};
}

// aggregates windows into bounding boxes
// img_size is needed to make sure the bounding box isn't out of bound
std::vector<bbox> aggregate_windows(const int img_w, const int img_h,
                                    const std::vector<window> &windows)
{
  // TODO
  return {};
}
