#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "mblbp.hh"
#include "window.hh"

struct weak_classifier
{
  weak_classifier(mblbp_feature feature) : feature(feature) {}
  mblbp_feature feature;
  double regression_parameters[255];
};

struct strong_classifier
{
  bool classify(const std::vector<mblbp_feature> &mblbp_features) const;
  std::vector<weak_classifier> weak_classifiers;
};

struct mblbp_classifier
{
  bool classify(const cv::Mat &integral, const window &potential_window) const;
  std::vector<strong_classifier> strong_classifiers;
};

// load a previously learned classifier from a file
mblbp_classifier load_classifier(const std::string &path);
