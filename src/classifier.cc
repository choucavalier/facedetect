#include <fstream>
#include <cstdlib>
#include <string>

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
  sum += this->sl;
  if ( sum > 0.0 )
    return true;
  return false;
}

double weak_classifier::evaluate(const cv::Mat &integral,
                                 const window &potential_window) const
{
  unsigned char feature_value = mblbp_calculate_feature(integral,
                                                        potential_window,
                                                        this->feature);
  return this->regression_parameters[feature_value];
}

void save_classifier(const mblbp_classifier &cascade,
                     const std::string &output_path)
{
  std::ofstream ofs(output_path);

  ofs << cascade.gamma_0 << std::endl
      << cascade.gamma_l << std::endl
      << cascade.beta_l << std::endl;

  ofs << initial_window_w << std::endl
      << initial_window_h << std::endl
      << cascade.strong_classifiers.size() << std::endl;
  for(const auto& strong_classifier : cascade.strong_classifiers)
  {
    ofs << strong_classifier.weak_classifiers.size() << std::endl;
    for(const auto& weak_classifier : strong_classifier.weak_classifiers)
    {
      ofs << weak_classifier.feature.x << std::endl
          << weak_classifier.feature.y << std::endl
          << weak_classifier.feature.block_w << std::endl
          << weak_classifier.feature.block_h << std::endl;
      for(int i = 0; i < 256; ++i)
        ofs << weak_classifier.regression_parameters[i] << std::endl;
      ofs << weak_classifier.k << std::endl;
    }
    ofs << strong_classifier.sl << std::endl;
  }

}

mblbp_classifier load_classifier(const std::string &classifier_path)
{
  std::cout << "loading classifier : " << classifier_path << std::endl;
  mblbp_classifier cascade;

  std::ifstream ifs(classifier_path);
  std::string line;

  std::getline(ifs, line);
  cascade.gamma_0 = std::stod(line);
  std::getline(ifs, line);
  cascade.gamma_l = std::stod(line);
  std::getline(ifs, line);
  cascade.beta_l = std::stod(line);

  std::getline(ifs, line);
  int initial_window_w = std::atoi(line.c_str());
  std::getline(ifs, line);
  int initial_window_h = std::atoi(line.c_str());
  std::getline(ifs, line);
  int n_strong_classifiers = std::atoi(line.c_str());

  std::vector<strong_classifier> strong_classifiers;
  for(int sc_idx = 0; sc_idx < n_strong_classifiers; ++sc_idx)
  {
    strong_classifier sc;
    std::getline(ifs, line);
    int n_weak_classifiers = std::atoi(line.c_str());
    for(int wc_idx = 0; wc_idx < n_weak_classifiers; ++wc_idx)
    {
      std::getline(ifs, line);
      int x = std::atoi(line.c_str());
      std::getline(ifs, line);
      int y = std::atoi(line.c_str());
      std::getline(ifs, line);
      int block_w = std::atoi(line.c_str());
      std::getline(ifs, line);
      int block_h = std::atoi(line.c_str());

      mblbp_feature feature(x, y, block_w, block_h);
      weak_classifier wc = weak_classifier(feature);
      for(int i = 0; i < 256; ++i)
      {
        std::getline(ifs, line);
        wc.regression_parameters[i] = std::stod(line);
      }

      std::getline(ifs, line);
      wc.k = std::atoi(line.c_str());
      sc.weak_classifiers.push_back(wc);
    }
    sc.n_weak_classifiers = n_weak_classifiers;
    std::getline(ifs, line);
    sc.sl = std::stod(line);
    cascade.strong_classifiers.push_back(sc);
  }
  ifs.close();

  return cascade;
}
