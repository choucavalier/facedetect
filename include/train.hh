#pragma once

#include "classifier.hh"

/* Train a classifier
**
** Parameters
** ----------
** positive_path : string
**     Path to the directory containing positive windows examples
**
** negative_path : string
**     Path to the directory containing negative windows examples
**
** Return
** ------
** classifier : mblbp_classifier
**     Trained classifier
*/
mblbp_classifier train(const std::string &positive_path,
                       const std::string &negative_path);
