#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mblbp.hh"
#include "window.hh"

class weak_classifier
{
public:
  weak_classifier(mblbp_feature feature) : feature(feature) {}
  double evaluate(const cv::Mat &integral,
                  const window &potential_window) const;
  mblbp_feature feature;
  double regression_parameters[255];
};

class strong_classifier
{
public:
  bool classify(const cv::Mat &integral, const window &potential_window) const;
  std::vector<weak_classifier> weak_classifiers;
};

class mblbp_classifier
{
public:
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
** classifier : const& mblbp_classifier
**     The classifier to be saved
**
** output_path : const& std::string
**     Path to the file in which the classifier should be saved
*/
void save_classifier(const mblbp_classifier &classifier,
                     const std::string &output_path);
// load a previously learned classifier from a file
mblbp_classifier load_classifier(const std::string &classifier_path);
