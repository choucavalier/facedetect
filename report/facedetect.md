# Principle

A face detection algorithm looks at *features* of an image to determine whether
it is a face or not. In the Image Processing domain, they are called *visual
descriptors*.

## MB-LBP visual descriptors

MB-LBP of different sizes and at different locations are considered

\begin{cppcode}
std::vector<mblbp_feature> mblbp_all_features()
{
  std::vector<mblbp_feature> features;

  for(int block_w = min_block_size; block_w <= max_block_size; block_w += 3)
    for(int block_h = min_block_size; block_h <= max_block_size; block_h += 3)
      for(int x = 0; x <= initial_window_w - block_w; ++x)
        for(int y = 0; y <= initial_window_h - block_h; ++y)
          features.push_back(mblbp_feature(x, y, block_w, block_h));

  return features;
}
\end{cppcode}

## Feature selection using Gentle Adaboost

# Detection procedure

## Sliding window

The sliding window model is conceptually simple: **independently** classify all
image patches as being object or non-object. Sliding window classification is
the dominant paradigm in object detection and for one object category in
particular -- faces -- it is one of the most noticeable successes of computer
vision.

The following function generates a \mintinline{cpp}{std::vector} containing all
potential windows in an image.

\begin{cppcode}
std::vector<window> get_potential_windows(const int img_w, const int img_h)
{
  std::vector<window> potential_windows;

  double w = initial_window_w; // current window width
  double h = initial_window_h; // current window height
  double scale = 1.0; // current scale
  double shift = scale * shift_delta; // current window shift (in pixels)

  while(w <= max_window_w && h <= max_window_h)
  {
    for(double x = 0.0; x < static_cast<double>(img_w) - w; x += shift)
    {
      for(double y = 0.0; y < static_cast<double>(img_h) - h; y += shift)
        potential_windows.push_back(window(x, y, w, h, scale));
    }

    // update values for next loop
    w = w * scaling_factor;
    h = h * scaling_factor;
    scale = scale * scaling_factor;
    shift = scale * shift_delta;
  }

  return potential_windows;
}
\end{cppcode}

## Cascade of classifiers

# Classifier training process

## Data

### Dataset

### Preprocessing

## Gentle Adaboost
