#include <opencv2/opencv.hpp>

#include "classifier.hh"
#include "mblbp.hh"

bool mblbp_classifier::classify(const cv::Mat &integral,
                                const window &potential_window) const
{
  for(const auto& strong_classifier : this->strong_classifiers)
  {
    bool positive = strong_classifier.classify(integral, potential_window);
    if(!positive)
      return false;
  }

  return true;
}

bool strong_classifier::classify(const cv::Mat &integral,
                                 const window &potential_window) const
{
  double sum = 0;
  for(const auto& weak_classifier : this->weak_classifiers)
    sum += weak_classifier.evaluate(integral, potential_window);
  return sum > 0;
}

double weak_classifier::evaluate(const cv::Mat &integral,
                                 const window &potential_window) const
{
  int feature_val = mblbp_calculate_feature(integral, potential_window,
                                            this->feature);
}

mblbp_classifier load_classifier(const std::string &path)
{
  mblbp_classifier classifier;

  // TODO

  return classifier;
}
