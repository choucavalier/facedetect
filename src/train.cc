#include <random>
#include <utility>
#include <algorithm>
#include <tuple>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include "train.hh"
#include "window.hh"
#include "mblbp.hh"
#include "params.hh"

namespace fs = std::experimental::filesystem;
using data_t = std::vector<std::pair<std::vector<unsigned char>, char>>;

/* Generate all MB-LBP features inside a window
** The size of the window is defined in "parameters.hh"
**
** Return
** ------
** features : std::vector<mblbp_feature>
**     All features contained in a window
*/
static std::vector<mblbp_feature> mblbp_all_features()
{
  std::vector<mblbp_feature> features;
  for(int block_w = min_block_size; block_w <= max_block_size; block_w += 3)
    for(int block_h = min_block_size; block_h <= max_block_size; block_h += 3)
      for(int x = 0; x <= initial_window_w - block_w; ++x)
        for(int y = 0; y <= initial_window_h - block_h; ++y)
          features.push_back(mblbp_feature(x, y, block_w, block_h));
  return features;
}

static std::vector<unsigned char> mblbp_calculate_all_features(
  const cv::Mat &integral, const std::vector<mblbp_feature> &all_features)
{
  int n_features = all_features.size();
  std::vector<unsigned char> calculated_features(n_features);
  window base_window(0, 0, integral.rows, integral.cols, 1.0);
  for(int i = 0; i < n_features; ++i)
    calculated_features[i] = mblbp_calculate_feature(integral, base_window,
                                                     all_features[i]);

  return calculated_features;
}

static data_t load_data(const std::vector<mblbp_feature> &all_features,
                        const std::string &positive_path,
                        const std::string &negative_path)
{
  data_t data;
  cv::Mat img, integral;

  for(auto& directory_entry : fs::directory_iterator(positive_path))
  {
    std::string path = directory_entry.path();
    img = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::integral(img, integral);
    std::vector<unsigned char> features = mblbp_calculate_all_features(
      integral, all_features);
    data.push_back(std::make_pair(features, 1));
  }

  for(auto& directory_entry : fs::directory_iterator(negative_path))
  {
    std::string path = directory_entry.path();
    img = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::integral(img, integral);
    std::vector<unsigned char> features = mblbp_calculate_all_features(
      integral, all_features);
    data.push_back(std::make_pair(features, -1));
  }

  return data;
}

static void shuffle_data(data_t &data)
{
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);
}

static std::tuple<double, double, double, double> evaluate(
  const mblbp_classifier &classifier, const data_t &validation_set)
{
  int n_samples = validation_set.size();
  if(n_samples == 0)
    return std::make_tuple(0.0, 0.0, 0.0, 0.0);
  // tp: true positive, fn: false negative, etc.
  int n_tp = 0, n_tn = 0, n_fp = 0, n_fn = 0;
  char real_label, classification_label;

  for(int i = 0; i < n_samples; ++i)
  {
    real_label = validation_set[i].second;
    // calculate classification_label
    classification_label = 1;
    for(const auto& sc : classifier.strong_classifiers)
    {
      double sum = 0;
      for(const auto& wc : sc.weak_classifiers)
      {
        unsigned char feature_value = validation_set[i].first[wc.k];
        sum += wc.regression_parameters[feature_value];
      }
      if(sum < 0)
      {
        classification_label = -1;
        break;
      }
    }

    if(real_label == 1)
    {
      if(classification_label == 1)
        n_tp++;
      else
        n_fn++;
    }
    else
    {
      if(classification_label == -1)
        n_tn++;
      else
        n_fn++;
    }
  }

  double tp_rate = (double)n_tp / n_samples; // true positive rate
  double tn_rate = (double)n_tn / n_samples; // true negative rate
  double fp_rate = (double)n_fp / n_samples; // false positive rate
  double fn_rate = (double)n_fn / n_samples; // false negative rate

  auto rates = std::make_tuple(tp_rate, tn_rate, fp_rate, fn_rate);

  return rates;
}

mblbp_classifier train(const std::string &positive_path,
                       const std::string &negative_path)
{
  std::cout << std::string(10, '-') << std::endl;
  std::cout << "start training" << std::endl;
  std::cout << std::string(10, '-') << std::endl;
  std::cout << "using " << initial_window_w << "x" << initial_window_h
            << " windows" << std::endl;

  mblbp_classifier classifier;

  // retrieve all features for the configured initial window size
  auto all_features = mblbp_all_features();
  int n_features = all_features.size();
  std::cout << n_features << " features / image ("
            << sizeof(unsigned char) * n_features << " bytes)"
            << std::endl;
  // construct one weak_classifier per feature
  std::vector<weak_classifier> all_weak_classifiers;
  for(int k = 0; k < n_features; ++k)
    all_weak_classifiers.push_back(weak_classifier(all_features[k], k));

  data_t data = load_data(all_features, positive_path, negative_path);
  shuffle_data(data);

  std::cout << data.size() << " samples in dataset" << std::endl;

  std::size_t split_idx = 2 * data.size() / 3;
  data_t training_set(data.begin(), data.begin() + split_idx);
  data_t validation_set(data.begin() + split_idx, data.end());

  std::cout << training_set.size() << " samples in training set" << std::endl;
  std::cout << validation_set.size() << " samples in validation set"
            << std::endl;

  // weights initialization to 1 / N
  std::vector<double> weights(training_set.size());
  std::fill_n(weights.begin(), training_set.size(), 1.0 / training_set.size());

  double detection_rate, tp_rate, ng_rate, fp_rate, fn_rate;
  do
  {
    strong_classifier new_strong_classifier;

    for(int i = 0; i < train_n_weak_per_strong; ++i)
    {
      // update all weak classifiers regression parameters
      // calculate weighted square error for each weak_classifier
      // TODO
      // select best weak_classifier
      // TODO int best_idx = ???
      //weak_classifier best_weak_classifier = all_weak_classifiers[best_idx];
      // delete selected weak_classifier from the whole set
      //all_weak_classifiers.erase(all_weak_classifiers.begin() + best_idx);
      // add new weak_classifier to the strong_classifier
      //new_strong_classifier.weak_classifiers.push_back(best_weak_classifier);
    }

    // add new strong_classifier to the mblbp_classifier
    classifier.strong_classifiers.push_back(new_strong_classifier);

    // calculate new detection and miss rates
    std::tie(tp_rate, ng_rate, fp_rate, fn_rate) = evaluate(classifier,
                                                            validation_set);
    detection_rate = tp_rate + ng_rate;

  } while(classifier.strong_classifiers.size() < train_n_strong &&
          (detection_rate < target_detection_rate || fp_rate > target_fp_rate));

  return classifier;
}
