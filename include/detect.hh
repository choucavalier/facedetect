#pragma once

#include <string>
#include <vector>

#include "window.hh"

/*
** Detect bounding boxes around faces in the given image
*/
std::vector<bbox> detect(const std::string &img_path,
                         const std::string &classifier_path);
