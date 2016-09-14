#include <opencv2/opencv.hpp>

#include "classification.hh"
#include "mblbp.hh"

bool mblbp_classifier::classify(const cv::Mat &integral,
                                const window &potential_window) const
{
  std::vector<mblbp_feature> features = mblbp_extract_features(
    integral, potential_window);

  for(const auto& strong_classifier : this->strong_classifiers)
  {
    bool positive = strong_classifier.classify(features);
    if(!positive)
      return false;
  }

  return true;
}

bool strong_classifier::classify(const std::vector<mblbp_feature> &features) const
{
  double sum = 0;
  for(const auto& weak_classifier : this->weak_classifiers)
    sum += weak_classifier.regression_parameters[weak_classifier.feature.id];
  return sum > 0;
}

mblbp_classifier load_classifier(const std::string &path)
{
  mblbp_classifier classifier;

  // TODO

  return classifier;
}
