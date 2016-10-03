#!/usr/bin/env python3

import os
import sys
import subprocess

CMAKELISTS_STR = """cmake_minimum_required (VERSION 2.6)
project (facedetect)

find_package(OpenCV REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -lstdc++fs -Wall -W -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++14 -lstdc++fs -Wall -W -pedantic")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include_directories(include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()"""

DETECTION_STR = """# detection
add_executable(detect entry_point/detect.cc
                      src/window.cc
                      src/mblbp.cc
                      src/classifier.cc
                      src/detect.cc)
target_link_libraries(detect ${OpenCV_LIBS})"""

TRAINING_STR = """# training
add_executable(train entry_point/train.cc
                     src/train.cc
                     src/classifier.cc
                     src/mblbp.cc
                     src/window.cc)
target_link_libraries(train ${OpenCV_LIBS})
target_link_libraries(train stdc++fs)"""

PREPROCESSING_STR = """# dataset preprocessing
add_executable(preprocess entry_point/preprocess.cc)
target_link_libraries(preprocess ${OpenCV_LIBS})
target_link_libraries(preprocess stdc++fs)"""

TESTS_STR = """# tests
add_executable(test_io test/test_io.cc
                       src/mblbp.cc
                       src/classifier.cc
                       src/window.cc)"""

def main():
    output_str = CMAKELISTS_STR
    build_options = []

    if len(sys.argv) == 1:
        output_str = add_all(output_str, build_options)
    else:

        for arg in sys.argv[1:]:
            if arg not in ["detection", "training", "preprocessing", "tests"]:
                print("Error : unknown argument " + str(arg))
                sys.exit(1)

        if "detection" in sys.argv:
            output_str = add_detection(output_str, build_options)
        if "training" in sys.argv:
            output_str = add_training(output_str, build_options)
        if "preprocessing" in sys.argv:
            output_str = add_preprocessing(output_str, build_options)
        if "tests" in sys.argv:
            output_str = add_tests(output_str, build_options)

    output_str += "\n"

    cmakelists_file = open('CMakeLists.txt', 'w+')
    cmakelists_file.write(output_str)
    cmakelists_file.close()

    os.putenv("CXX", "/usr/bin/gcc-6.2/bin/g++")
    subprocess.check_call(["rm", "-rf", "build"])
    subprocess.check_call(["mkdir", "build"])
    os.chdir("build")
    subprocess.check_call(["cmake", ".."])
    subprocess.check_call(["make"])

    build_options_file = open('build_options.txt', 'w+')
    for opt in build_options:
        build_options_file.write(opt + "\n")
    build_options_file.close()


def add_detection(output_str, build_options):
    output_str += "\n\n"
    output_str += DETECTION_STR
    build_options.append("detection")
    return output_str

def add_training(output_str, build_options):
    output_str += "\n\n"
    output_str += TRAINING_STR
    build_options.append("training")
    return output_str

def add_preprocessing(output_str, build_options):
    output_str += "\n\n"
    output_str += PREPROCESSING_STR
    build_options.append("preprocessing")
    return output_str

def add_tests(output_str, build_options):
    output_str += "\n\n"
    output_str += TESTS_STR
    build_options.append("tests")
    return output_str

def add_all(output_str, build_options):
    output_str = add_detection(output_str, build_options)
    output_str = add_training(output_str, build_options)
    output_str = add_preprocessing(output_str, build_options)
    #output_str = add_tests(output_str, build_options)
    return output_str

if __name__ == '__main__':
    main()
