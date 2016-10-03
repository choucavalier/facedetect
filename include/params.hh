#pragma once

// default parameters used to detect faces
const int initial_window_w = 20;
const int initial_window_h = 20;
const int max_window_w = 200;
const int max_window_h = 200;
const int min_block_size = 3;
const int max_block_size = 12;
const double scaling_factor = 1.25;
const double shift_delta = 1.5;

// training parameters
const double target_detection_rate = 0.95;
const double target_fp_rate = 0.01;
// number of weak classifiers to select per strong classifier
const int train_n_weak_per_strong = 10;

const int max_iteration_train_set_size = 1000;

// desired overall false positive rate
const double gamma_0 = 0.01;

// targeted layer false positive
const double gamma_l = 0.5;

// targeted layer false negative
const double beta_l = 0.005;
