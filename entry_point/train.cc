#include <chrono>
#include <iostream>

#include "classifier.hh"
#include "train.hh"

int main(int argc, char **argv)
{
  if(argc < 4)
  {
    std::cout << "usage: ./train data/positives data/negatives classifier.txt"
              << std::endl;
    return 1;
  }

  std::string positive_path(argv[1]);
  std::string negative_path(argv[2]);
  std::string output_path(argv[3]);

  using steady_clock = std::chrono::steady_clock;

  steady_clock::time_point begin = steady_clock::now();

  mblbp_classifier classifier = train(positive_path, negative_path);

  steady_clock::time_point end = steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin);
  int elapsed = duration.count();
  std::cout << "learning time: " << elapsed << "s" << std::endl;

  std::cout << "saving classifier in " << output_path << std::endl;

  save_classifier(classifier, output_path);
}
