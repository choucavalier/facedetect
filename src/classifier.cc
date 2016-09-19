#include <fstream>

#include <opencv2/opencv.hpp>

#include "classifier.hh"
#include "mblbp.hh"
#include "params.hh"

bool mblbp_classifier::classify(const cv::Mat &integral,
                                const window &potential_window) const
{
  for(const auto& strong_classifier : this->strong_classifiers)
    if(!strong_classifier.classify(integral, potential_window))
      return false;

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
  int feature_value = mblbp_calculate_feature(integral, potential_window,
                                              this->feature);
  return this->regression_parameters[feature_value];
}

void save_classifier(const mblbp_classifier &classifier,
                     const std::string &output_path)
{
  std::ofstream ofs(output_path);

  ofs << initial_window_w << std::endl
        << initial_window_h << std::endl
        << classifier.strong_classifiers.size() << std::endl;
  for(const auto& strong_classifier : classifier.strong_classifiers)
  {
    ofs << strong_classifier.weak_classifiers.size() << std::endl;
    for(const auto& weak_classifier : strong_classifier.weak_classifiers)
    {
      ofs << weak_classifier.feature.x << std::endl
         << weak_classifier.feature.y << std::endl
         << weak_classifier.feature.block_w << std::endl
         << weak_classifier.feature.block_h << std::endl;
      for(int i = 0; i < 255; ++i)
        ofs << weak_classifier.regression_parameters[i] << std::endl;
    }
  }
}

mblbp_classifier load_classifier(const std::string &classifier_path)
{
  mblbp_classifier classifier;

  std::ifstream ifs(classifier_path);

  int initial_window_w, initial_window_h;
  ifs >> initial_window_w >> initial_window_h;
  int n_strong_classifiers;
  ifs >> n_strong_classifiers;
  for(int sc_idx = 0; sc_idx < n_strong_classifiers; ++sc_idx)
  {
    strong_classifier sc;
    int n_weak_classifiers;
    ifs >> n_weak_classifiers;
    for(int wc_idx = 0; wc_idx < n_weak_classifiers; ++wc_idx)
    {
      int x, y, block_w, block_h;
      ifs >> x
          >> y
          >> block_w
          >> block_h;
      mblbp_feature feature(x, y, block_w, block_h);
      weak_classifier wc = weak_classifier(feature);
      for(int i = 0; i < 255; ++i)
        ifs >> wc.regression_parameters[i];
      sc.weak_classifiers.push_back(wc);
    }
    classifier.strong_classifiers.push_back(sc);
  }

  return classifier;
}
