#include <random>
#include <utility>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include "train.hh"
#include "window.hh"
#include "mblbp.hh"
#include "params.hh"

namespace fs = std::experimental::filesystem;
namespace chrono = std::chrono;
using data_t = std::vector<std::pair<std::vector<unsigned char>, char>>;

/* Estimate the memory used by a data set
**
** Paramaters
** ----------
** data : data_t
**     Dataset where each sample is represented as a vector of features
**
** Return
** ------
** size_in_bytes : long
**     Estimated size of the dataset in bytes
*/
long estimate_memory_size(data_t data)
{
  int n_samples = data.size();
  int n_features = data[0].first.size();

  return n_samples * n_features * sizeof(unsigned char);
}

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
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  data_t data;

  std::vector<std::string> positive_paths;
  std::vector<std::string> negative_paths;

  for(auto& directory_entry : fs::directory_iterator(positive_path))
    positive_paths.push_back(directory_entry.path().string());

  std::cout << positive_paths.size() << " positive samples" << std::endl;

  #pragma omp parallel for
  //for(std::size_t i = 0; i < 100; ++i)
  for(std::size_t i = 0; i < positive_paths.size(); ++i)
  {
    cv::Mat img, integral;
    img = cv::imread(positive_paths[i], CV_LOAD_IMAGE_GRAYSCALE);
    cv::integral(img, integral);
    std::vector<unsigned char> features = mblbp_calculate_all_features(
      integral, all_features);
    #pragma omp critical(add_features_positive)
    {
      data.push_back(std::make_pair(features, 1));
    }
  }

  for(auto& directory_entry : fs::directory_iterator(negative_path))
  {
    negative_paths.push_back(directory_entry.path().string());
  }

  std::cout << negative_paths.size() << " negative samples" << std::endl;

  #pragma omp parallel for
  //for(std::size_t i = 0; i < 100; ++i)
  for(std::size_t i = 0; i < negative_paths.size(); ++i)
  {
    cv::Mat img, integral;
    img = cv::imread(negative_paths[i], CV_LOAD_IMAGE_GRAYSCALE);
    cv::integral(img, integral);
    std::vector<unsigned char> features = mblbp_calculate_all_features(
      integral, all_features);
    #pragma omp critical(add_features_negative)
    {
      data.push_back(std::make_pair(features, -1));
    }
  }

  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  auto duration = chrono::duration_cast<chrono::seconds>(end - begin);
  int elapsed = duration.count();
  std::cout << "loading data took " << elapsed << "s" << std::endl;

  return data;
}

static data_t select_data_cascade(const data_t& training_set,
                                  const mblbp_classifier &classifier)
{
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  data_t positive_training_examples;
  data_t negative_training_examples;

  size_t n_pos = 0;
  size_t n_neg = 0;

  char classification_label;

  for (const auto& training_example : training_set)
  {
    classification_label = 1;
    for(const auto& sc : classifier.strong_classifiers)
    {
      double sum = 0;
      for(const auto& wc : sc.weak_classifiers)
      {
        int feature_value = training_example.first[wc.k];
        sum += wc.regression_parameters[feature_value];
      }
      if(sum < 0)
      {
        classification_label = -1;
        break;
      }
    }

    if (training_example.second == 1)
      ++n_pos;
    else
      ++n_neg;

    if (classification_label == 1)
    {
      if (training_example.second == 1)
        positive_training_examples.push_back(training_example);
      else
        negative_training_examples.push_back(training_example);
    }
  }

  std::cout << "positive training examples passed through classification : "
            << positive_training_examples.size();
  std::cout << " / " << n_pos << std::endl;
  std::cout << "negative training examples passed through classification : "
            << negative_training_examples.size();
  std::cout << " / " << n_neg << std::endl;

  if (positive_training_examples.size() < negative_training_examples.size())
    negative_training_examples.resize(positive_training_examples.size());
  if (negative_training_examples.size() < positive_training_examples.size())
    positive_training_examples.resize(negative_training_examples.size());

  std::cout << "selected number of positive training examples : "
            << positive_training_examples.size() << std::endl;
  std::cout << "selected number of negative training examples : "
            << negative_training_examples.size() << std::endl;

  data_t selected_data;
  selected_data.insert(selected_data.end(), positive_training_examples.begin(),
                       positive_training_examples.end());
  selected_data.insert(selected_data.end(), negative_training_examples.begin(),
                       negative_training_examples.end());

  return selected_data;
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

  int n_positive = 0, n_negative = 0;

  for(int i = 0; i < n_samples; ++i)
  {
    real_label = validation_set[i].second;
    if(real_label == 1)
      n_positive++;
    else
      n_negative++;
    // calculate classification_label
    classification_label = 1;
    for(const auto& sc : classifier.strong_classifiers)
    {
      double sum = 0;
      for(const auto& wc : sc.weak_classifiers)
      {
        int feature_value = validation_set[i].first[wc.k];
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
        n_fp++;
    }
  }

  double tp_rate = (double)n_tp / n_positive; // true positive rate
  double tn_rate = (double)n_tn / n_negative; // true negative rate
  double fp_rate = (double)n_fp / n_negative; // false positive rate
  double fn_rate = (double)n_fn / n_positive; // false negative rate

  auto rates = std::make_tuple(tp_rate, tn_rate, fp_rate, fn_rate);

  return rates;
}

void remove_true_and_false_negatives(const mblbp_classifier &classifier,
                                     data_t &training_set)
{
  char classification_label;

  auto iter = training_set.begin();
  while (iter != training_set.end())
  {
    classification_label = 1;
    for(const auto& sc : classifier.strong_classifiers)
    {
      double sum = 0;
      for(const auto& wc : sc.weak_classifiers)
      {
        int feature_value = iter->first[wc.k];
        sum += wc.regression_parameters[feature_value];
      }
      if(sum < 0)
      {
        classification_label = -1;
        break;
      }
    }

    if (classification_label == -1)
      iter = training_set.erase(iter);
    else
      ++iter;
  }
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

  std::cout << std::string(10, '-') << std::endl;

  std::cout << "calculating all MB-LBP features on all samples" << std::endl;
  data_t data = load_data(all_features, positive_path, negative_path);
  std::cout << "shuffling data of size : " << data.size() << std::endl;
  shuffle_data(data);

  std::cout << data.size() << " samples in dataset" << std::endl;
  std::cout << estimate_memory_size(data) << " bytes in memory" << std::endl;

  // split dataset
  std::size_t split_idx = 4 * data.size() / 5;
  data_t training_set(data.begin(), data.begin() + split_idx);
  data_t validation_set(data.begin() + split_idx, data.end());

  std::cout << training_set.size() << " samples in training set" << std::endl;
  std::cout << validation_set.size() << " samples in validation set"
            << std::endl;

  std::cout << std::string(10, '-') << std::endl;

  // current rates on validation set
  double detection_rate, tp_rate, tn_rate, fp_rate, fn_rate;

  int n_iter = 1;
  size_t n_negative_examples = 0;
  do
  {
    std::cout << "iteration #" << n_iter << std::endl;

    data_t iteration_training_set = select_data_cascade(training_set, classifier);

    // weights initialization to 1 / n_samples
    std::vector<double> weights(iteration_training_set.size());
    std::fill_n(weights.begin(), iteration_training_set.size(), 1.0 / iteration_training_set.size());

    strong_classifier new_strong_classifier;

    // select train_n_weak_per_strong weak classifiers
    for(int n_weak = 0; n_weak < train_n_weak_per_strong; ++n_weak)
    {
      int best_idx = -1; // index of the best weak_classifier (smallest wse)
      double best_wse; // current smallest (best) wse, used to update best_idx
      // update all weak classifiers regression parameters
      #pragma omp parallel for
      for(std::size_t wc_idx = 0 ; wc_idx < all_weak_classifiers.size();
          wc_idx++)
      {
        weak_classifier &wc = all_weak_classifiers[wc_idx];
        for(int j = 0; j < 256; ++j)
        {
          double numerator = 0, denominator = 0;
          for(std::size_t i = 0; i < iteration_training_set.size(); ++i)
          {
            if(iteration_training_set[i].first[wc.k] == j)
            {
              numerator += weights[i] * iteration_training_set[i].second;
              denominator += weights[i];
            }
          }
          if(denominator != 0)
            wc.regression_parameters[j] = numerator / denominator;
          else
            wc.regression_parameters[j] = 0.5;
        }
        double wse = 0;
        for(std::size_t i = 0; i < iteration_training_set.size(); ++i)
        {
          double value = wc.regression_parameters[iteration_training_set[i].first[wc.k]];
          wse += weights[i] * std::pow(value - iteration_training_set[i].second, 2);
        }
        #pragma omp critical(best_wse_update)
        {
          if(best_idx < 0 || wse < best_wse)
          {
            best_wse = wse;
            best_idx = wc_idx;
          }
        }
      }
      std::cout << "best wse " << best_wse << "  " << n_weak << " / " << train_n_weak_per_strong << std::endl;
      // get a copy of the best weak_classifier before deleting it
      weak_classifier best_weak_classifier(all_weak_classifiers[best_idx]);

      weak_classifier &bwc = best_weak_classifier;
      //std::cout << "new weak_classifier:" << std::endl;
      //std::cout << "    feature:" << std::endl;
      //std::cout << "        (x, y) = (" << bwc.feature.x << ", "
                                        //<< bwc.feature.y << ")" << std::endl;

      //std::cout << "        block_w = " << bwc.feature.block_w << std::endl;
      //std::cout << "        block_h = " << bwc.feature.block_h << std::endl;
      //std::cout << "    k = " << bwc.k << std::endl;
      //std::cout << "    regression_parameters = [ ";
      //for(int i = 0; i < 256; ++i)
        //std::cout << bwc.regression_parameters[i] << " ";
      //std::cout << "]" << std::endl;

      // delete selected weak_classifier from the whole set
      all_weak_classifiers.erase(all_weak_classifiers.begin() + best_idx);
      // add new weak_classifier to the strong_classifier
      new_strong_classifier.weak_classifiers.push_back(best_weak_classifier);

      // update weights
      double sum = 0;
      for(std::size_t i = 0; i < weights.size(); ++i)
      {
        double value = bwc.regression_parameters[iteration_training_set[i].first[bwc.k]];
        weights[i] = weights[i] * std::exp(-iteration_training_set[i].second * value);
        sum += weights[i];
      }
      for(std::size_t i = 0; i < weights.size(); ++i)
        weights[i] /= sum;


      /*
      // calculate new detection and miss rates
      std::tie(tp_rate, tn_rate, fp_rate, fn_rate) = evaluate(classifier,
                                                              validation_set);
      detection_rate = tp_rate;



      std::cout << "detection_rate = " << detection_rate << std::endl;
      std::cout << "true positive = " << tp_rate << std::endl;
      std::cout << "true negative = " << tn_rate << std::endl;
      std::cout << "false positive = " << fp_rate << std::endl;
      std::cout << "false negative = " << fn_rate << std::endl;
      std::cout << "best_wse = " << best_wse << std::endl;
      std::cout << "training set size = " << training_set.size() << std::endl;
      std::cout << std::string(10, '-') << std::endl;
      */
    }

    // add new strong_classifier to the mblbp_classifier
    classifier.strong_classifiers.push_back(new_strong_classifier);

    // calculate new detection and miss rates
    std::tie(tp_rate, tn_rate, fp_rate, fn_rate) = evaluate(classifier,
                                                            validation_set);
    detection_rate = tp_rate;

    remove_true_and_false_negatives(classifier, training_set);

    n_negative_examples = 0;
    for (auto iter : iteration_training_set)
    {
      if (iter.second != 1)
        ++n_negative_examples;
    }

    std::cout << "detection_rate = " << detection_rate << std::endl;
    std::cout << "true positive = " << tp_rate << std::endl;
    std::cout << "true negative = " << tn_rate << std::endl;
    std::cout << "false positive = " << fp_rate << std::endl;
    std::cout << "false negative = " << fn_rate << std::endl;
    std::cout << "training set size = " << iteration_training_set.size() << std::endl;
    std::cout << "negative examples in training set : " << n_negative_examples << std::endl;

    n_iter++;

  } while(classifier.strong_classifiers.size() < train_n_strong &&
          (detection_rate < target_detection_rate || fp_rate > target_fp_rate ) && n_negative_examples > 0 );

  return classifier;
}
