#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mblbp.hh"
#include "window.hh"

struct weak_classifier
{
  weak_classifier(const mblbp_feature &feature, const int k = 0) :
    feature(feature), k(k) {}

  /* Calculate the weak_classifier value on a potential_window
  **
  ** Parameters
  ** ----------
  ** integral : cv::Mat
  **     Integral image
  **
  ** potential_window : window
  **     Window on which to evaluate the weak_classifier
  **
  ** Return
  ** ------
  ** regression_parameter : double
  **     The learned regression parameter for feature_value
  */
  double evaluate(const cv::Mat &integral,
                  const window &potential_window) const;

  mblbp_feature feature;
  double regression_parameters[255];
  int k;
};

struct strong_classifier
{
  /* Classify a window
  **
  ** Parameters
  ** ----------
  ** integral : cv::Mat
  **     Integral image
  **
  ** potential_window : window
  **     Window to classify as containing a face or not
  **
  ** Return
  ** ------
  ** positive : bool
  **     Whether the window contains a face or not
  */
  bool classify(const cv::Mat &integral, const window &potential_window) const;

  std::vector<weak_classifier> weak_classifiers;
};

struct mblbp_classifier
{
  /* Classify a window
  **
  ** Parameters
  ** ----------
  ** integral : cv::Mat
  **     Integral image
  **
  ** potential_window : window
  **     Window to classify as containing a face or not
  **
  ** Return
  ** ------
  ** positive : bool
  **     Whether the window contains a face or not
  */
  bool classify(const cv::Mat &integral, const window &potential_window) const;

  std::vector<strong_classifier> strong_classifiers;
};

/* Save a learned classifier to a file
**
** The classifier is saved in the following format:
**
** <initial_window_w>
** <initial_window_h>
** <n_strong_classifiers>
** ------ for every strong_classifier ---------------
** | <n_weak_classifiers>
**   ------ for every weak_classifier ---------------
**   | <weak_classifier.feature.x>
**   | <weak_classifier.feature.y>
**   | <weak_classifier.feature.block_w>
**   | <weak_classifier.feature.block_h>
**   | <weak_classifier.regression_parameters[0]>
**   | ...
**   | <weak_classifier.regression_parameters[254]>
** | ...
**
** Note that the `|` and the indendation are not included.
**
** Parameters
** ----------
** classifier : mblbp_classifier
**     The classifier to be saved
**
** output_path : std::string
**     Path to the file in which the classifier should be saved
*/
void save_classifier(const mblbp_classifier &classifier,
                     const std::string &output_path);

/* Load a classifier from a file
**
** Parameters
** ----------
** classifier_path : std::string
**     Path to the saved classifier
**
** Return
** ------
** classifier : mblbp_classifier
**     Loaded classifier
*/
mblbp_classifier load_classifier(const std::string &classifier_path);
