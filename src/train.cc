#include <random>
#include <utility>
#include <experimental/filesystem>

#include "train.hh"

namespace fs = std::experimental::filesystem;

static std::vector<std::pair<std::string, bool>> load_data(
  const std::string &positive_path, const std::string &negative_path)
{
  std::vector<std::pair<std::string, bool>> data;

  for(auto& directory_entry : fs::directory_iterator(positive_path))
    data.push_back(std::make_pair(directory_entry.path(), true));

  for(auto& directory_entry : fs::directory_iterator(negative_path))
    data.push_back(std::make_pair(directory_entry.path(), false));

  return data;
}

static void shuffle_data(std::vector<std::pair<std::string, bool>> &data)
{
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);
}

mblbp_classifier train(const std::string &positive_path,
                       const std::string &negative_path)
{
  mblbp_classifier classifier;

  auto data = load_data(positive_path, negative_path);
  shuffle_data(data);

  for(const auto& data_point : data)
    std::cout << data_point.first << " " << data_point.second << std::endl;

  return classifier;
}
