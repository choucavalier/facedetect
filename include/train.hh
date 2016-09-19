#pragma once

#include "classifier.hh"

/* Learn a classifier
**
** Parameters
** ----------
** positive_path : string
**     Path to the directory containing positive windows examples
**
** negative_path : string
**     Path to the directory containing negative windows examples
*/
mblbp_classifier train(const std::string &positive_path,
                       const std::string &negative_path);
