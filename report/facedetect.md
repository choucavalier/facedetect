# Principle

A face detection algorithm looks at *features* of an image to determine whether
it is a face or not. In the Image Processing domain, they are called *visual
descriptors*. Multi-block Local Binary Patterns (MB-LBP) is one type of visual
descriptor that can be used for detecting faces (and more generally, objects)
in images. \newline

It is important to note that the detection  is developed for a very
specific window size, say $20 \times 20$ pixels, and this window is then
shifted and scaled in the detection procedure in order to detect faces of
different sizes and at different locations in the image. We will refer to this
unshifted and unscaled window as the *window of reference*.

## Multi-block Local Binary Patterns

Traditional Haar-like rectangle features measure the difference between the
average intensities of rectangular regions. For example, the value of
a two-rectangle filter is the difference between the sums of the pixels within
two rectangular regions. If we change the position, size, shape and arrangement
of rectangular regions, the Haar-like features can capture the intensity
gradient at different locations, spatial frequencies and directions. Viola an
Jones \cite{viola} applied three kinds of such features for detecting frontal
faces. By using the integral image, any rectangle filter type, at any scale or
location, can be evaluated in constant time. However, the Haar-like features
seem too simple and show some limits. \newline


MB-LBP is an extension of LBP \cite{lbp} that can be computed on multiple
scales in constant time using the integral image. 9 equally-sized rectangles
are used to compute a feature. For each rectangle, the sum of the pixel
intensities is computed. Comparisons of these sums to that of the central
rectangle determine the feature, similarly to LBP.

\begin{figure}[H]
\begin{center}
\includegraphics[scale=0.7]{lbp.png}
\caption{Local Binary Patterns ($3\times3$ MB-LBP)}
\end{center}
\end{figure}

The output of the MB-LBP operator can be obtained as follows:

$$\text{MB-LBP} = \sum_{i=1}^8 \text{sign}(s_i - s_c)2^i$$

where $s_i$ is the sum of the pixel intensities of the $i$th neighborhood
rectangle, $s_c$ is the sum of the pixel intensities of the central rectangle,

$$\text{sign}(x) =
  \begin{cases}
    \enskip 1 \quad \text{if } x > 0 \\
    \enskip 0 \quad \text{if } x \le 0
  \end{cases}$$

MB-LBP of different sizes and at different locations (inside the reference
window) are considered. The following function constructs
a \mintinline{cpp}{std::vector} containing all MB-LBP features inside
a considered window:

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

This section describes the different aspects of the detection process, used to
apply the trained classifier.

## Sliding window

The sliding window model is conceptually simple: **independently** classify all
image patches as being object or non-object. Sliding window classification is
the dominant paradigm in object detection and for one object category in
particular -- faces -- it is one of the most noticeable successes of computer
vision. \newline

The following function constructs a \mintinline{cpp}{std::vector} containing
all potential windows given the size of the image to be analyzed:

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
      for(double y = 0.0; y < static_cast<double>(img_h) - h; y += shift)
        potential_windows.push_back(window(x, y, w, h, scale));

    // update values for next loop
    w = w * scaling_factor;
    h = h * scaling_factor;
    scale = scale * scaling_factor;
    shift = scale * shift_delta;
  }

  return potential_windows;
}
\end{cppcode}

\mintinline{cpp}{scaling_factor} and \mintinline{cpp}{shift_delta}, as
described in \cite{viola}, are used to parametrize the sliding window process.

## Cascade of classifiers

# Classifier training process

## Data

### Dataset

### Preprocessing

## Gentle Adaboost
