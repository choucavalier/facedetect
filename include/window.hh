#pragma once

#include <vector>

// a window that might contain a face
struct window
{
  window(double x, double y, double w, double h, double scale)
    : x(x), y(y), w(w), h(h), scale(scale) {}
  // absolute offset
  double x;
  double y;
  // rectangle dimensions
  double w;
  double h;
  // scaling from the initial window size
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

/* Get all windows in the image that must be classified
**
** The classifier will be called for all of these windows
**
** Parameters
** ----------
** img_w, img_h : int
**     Width and height of the image, needed to make sure the windows ar not out
**     of bounds
**
** Return
** ------
** potential_windows : std::vector<window>
**     All windows to classify
*/
std::vector<window> get_potential_windows(int img_w, int img_h);

/* Aggregate positive windows into bounding boxes
**
** Parameters
** ----------
** img_w, img_h : int
**     Width and height of the image, needed to make sure the bounding boxes are
**     not out of bound
**
** windows : const& std::vector<window>
**     All positive windows found in the image
**
** Return
** ------
** bounding_boxes : std::vector<bbox>
**     Bounding boxes around faces detected in the image
*/
std::vector<bbox> aggregate_windows(int img_w, int img_h,
                                    const std::vector<window> &windows);
