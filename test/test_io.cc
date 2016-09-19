#include <iostream>
#include <cmath>
#include <random>

#include "classifier.hh"

weak_classifier random_weak_classifier()
{
  mblbp_feature feature(0, 0, 9, 9);
  return weak_classifier(feature);
}

strong_classifier random_strong_classifier()
{
  strong_classifier classifier;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist(20, 100);

  int n_wc = dist(rng); // number of weak_classifiers to generate

  for(int i = 0; i < n_wc; ++i)
    classifier.weak_classifiers.push_back(random_weak_classifier());

  return classifier;
}

mblbp_classifier random_classifier()
{
  mblbp_classifier classifier;

  int n_sc = 50; // number of strong_classifiers to generate

  for(int i = 0; i < n_sc; ++i)
    classifier.strong_classifiers.push_back(random_strong_classifier());

  return classifier;
}

bool double_eq(double a, double b)
{
  return std::abs(a - b) < 1e-3;
}

int main()
{
  std::cout << "test_io... ";

  mblbp_classifier classifier = random_classifier();

  save_classifier(classifier, "/tmp/classifier.txt");

  mblbp_classifier loaded = load_classifier("/tmp/classifier.txt");

  int n_sc = classifier.strong_classifiers.size();
  int loaded_n_sc = loaded.strong_classifiers.size();

  assert(loaded_n_sc == n_sc);

  for(int i = 0; i < n_sc; ++i)
  {
    int n_wc = classifier.strong_classifiers[i].weak_classifiers.size();
    int loaded_n_wc = loaded.strong_classifiers[i].weak_classifiers.size();

    assert(loaded_n_wc == n_wc);

    for(int j = 0; j < n_wc; ++j)
    {
      auto wc = classifier.strong_classifiers[i].weak_classifiers[j];
      auto loaded_wc = loaded.strong_classifiers[i].weak_classifiers[j];
      assert(wc.feature == loaded_wc.feature);
      for(int k = 0; k < 255; ++k)
        assert(double_eq(wc.regression_parameters[k],
                         loaded_wc.regression_parameters[k]));
    }
  }

  std::cout << "OK" << std::endl;
}
