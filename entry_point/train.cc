#include <chrono>
#include <iostream>

#include "classifier.hh"
#include "train.hh"
#include "params.hh"

namespace chrono = std::chrono;

int main(int argc, char **argv)
{
  if(argc < 3)
  {
    std::cout << "usage: ./train data/positives data/negatives [classifier.txt]"
              << std::endl;
    return 1;
  }

  std::string positive_path(argv[1]); // path to positive samples
  std::string negative_path(argv[2]); // path to negative samples

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  mblbp_classifier cascade;
  if ( argc == 4 )
  {
    std::cout << "loading classifier " << argv[3] << std::endl;
    std::string classifier_path(argv[3]); // where to load the classifier from
    cascade = load_classifier(classifier_path);
    std::cout << "cascade loaded" << std::endl;
    std::cout << "number of strong classifiers : " << cascade.strong_classifiers.size() << std::endl;
  }
  else
  {
    std::cout << "Creating new cascade" << std::endl;
    cascade.gamma_0 = gamma_0;
    cascade.gamma_l = gamma_l;
    cascade.beta_l = beta_l;
  }

  std::cout << "desired overall false positive rate : " << cascade.gamma_0 << std::endl;
  std::cout << "targeted layer false positive rate : " << cascade.gamma_l << std::endl;
  std::cout << "targeted layer false negative rate : " << cascade.beta_l << std::endl;
  train(cascade, positive_path, negative_path);

  chrono::steady_clock::time_point end = chrono::steady_clock::now();

  auto duration = chrono::duration_cast<chrono::seconds>(end - begin);
  int elapsed = duration.count();
  std::cout << "learning time: " << elapsed << "s" << std::endl;
}
