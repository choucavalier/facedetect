cmake_minimum_required (VERSION 2.6)
project (facedetect)

find_package(OpenCV REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -W -pedantic -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -W -pedantic -g")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include_directories(include)

# detection
#add_executable(detect entry_point/detect.cc
                      #src/window.cc
                      #src/mblbp.cc
                      #src/classifier.cc
                      #src/detect.cc)

# training
add_executable(train entry_point/train.cc
                     src/train.cc
                     src/classifier.cc
                     src/mblbp.cc
                     src/window.cc)

# tests
#add_executable(test_io test/test_io.cc
                       #src/mblbp.cc
                       #src/classifier.cc
                       #src/window.cc)

#target_link_libraries(detect ${OpenCV_LIBS})
target_link_libraries(train ${OpenCV_LIBS})
target_link_libraries(train stdc++fs)
