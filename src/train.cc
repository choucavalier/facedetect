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
using data_t = std::vector<std::pair<std::string, bool>>;

static data_t load_data(const std::string &positive_path,
                        const std::string &negative_path)
{
  std::vector<std::pair<std::string, bool>> data;

  for(auto& directory_entry : fs::directory_iterator(positive_path))
    data.push_back(std::make_pair(directory_entry.path(), true));

  for(auto& directory_entry : fs::directory_iterator(negative_path))
    data.push_back(std::make_pair(directory_entry.path(), false));

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
  int size = validation_set.size();
  // tp: true positive, fn: false negative, etc.
  int n_tp = 0, n_tn = 0, n_fp = 0, n_fn = 0;

  for(int i = 0; i < size; ++i)
  {
    std::string path = validation_set[i].first;
    bool real_label = validation_set[i].second;

    cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    int img_w = img.rows;
    int img_h = img.cols;
    cv::Mat integral;
    cv::integral(img, integral);
    window img_window(0, 0, img_w, img_h, 1.0);
    bool classification_label = classifier.classify(integral, img_window);

    if(real_label == true)
    {
      if(classification_label == true)
        n_tp++;
      else
        n_fn++;
    }
    else
    {
      if(classification_label == false)
        n_tn++;
      else
        n_fn++;
    }
  }

  double tp_rate = (double)n_tp / size; // true positive rate
  double tn_rate = (double)n_tn / size; // true negative rate
  double fp_rate = (double)n_fp / size; // false positive rate
  double fn_rate = (double)n_fn / size; // false negative rate

  auto rates = std::make_tuple(tp_rate, tn_rate, fp_rate, fn_rate);

  return rates;
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

mblbp_classifier train(const std::string &positive_path,
                       const std::string &negative_path)
{
  mblbp_classifier classifier;

  auto data = load_data(positive_path, negative_path);
  shuffle_data(data);

  for(const auto& data_point : data)
    std::cout << data_point.first << " " << data_point.second << std::endl;

  data_t training_set;
  data_t validation_set;

  // retrieve all features for the configured initial window size
  auto all_features = mblbp_all_features();
  int n_features = all_features.size();

  // weights initialization to 1 / N
  std::vector<double> weights(training_set.size());
  std::fill_n(weights.begin(), training_set.size(), 1.0 / training_set.size());

  double tp_rate, ng_rate, fp_rate, fn_rate;
  double detection_rate;
  do
  {
    std::tie(tp_rate, ng_rate, fp_rate, fn_rate) = evaluate(classifier,
                                                            validation_set);
    detection_rate = tp_rate + ng_rate;

    std::cout << "detection rate: " << detection_rate << std::endl;

  } while(detection_rate < target_detection_rate || fp_rate > target_fp_rate);

  return classifier;
}
