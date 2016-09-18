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
private:
  mblbp_feature feature;
  double regression_parameters[255];
};

class strong_classifier
{
public:
  bool classify(const cv::Mat &integral, const window &potential_window) const;
private:
  std::vector<weak_classifier> weak_classifiers;
};

class mblbp_classifier
{
public:
  bool classify(const cv::Mat &integral, const window &potential_window) const;
private:
  std::vector<strong_classifier> strong_classifiers;
};

// save a classifier
void save_classifier(const mblbp_classifier &classifier,
                     const std::string &output_path);
// load a previously learned classifier from a file
mblbp_classifier load_classifier(const std::string &classifier_path);
