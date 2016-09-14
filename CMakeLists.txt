cmake_minimum_required (VERSION 2.6)
project (facedetect)

find_package( OpenCV REQUIRED )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -W -pedantic -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -W -pedantic -g")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include_directories ( include )

add_executable(facedetect src/window.cc
                          src/mblbp.cc
                          src/classifier.cc
                          src/detect.cc
                          src/main.cc)

target_link_libraries(facedetect ${OpenCV_LIBS})
