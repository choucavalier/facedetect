#pragma once

// default parameters used to detect faces
const int initial_window_w = 20;
const int initial_window_h = 20;
const int max_window_w = 200;
const int max_window_h = 200;
const int min_block_size = 3;
const int max_block_size = 18;
const double scaling_factor = 1.25;
const double shift_delta = 1.5;

// training parameters
const double target_detection_rate = 0.92;
const double target_fp_rate = 0.01;
