#!/usr/bin/env python3

import os
import sys
import subprocess
import glob

def check_build():
    if not os.path.exists("build/"):
        return False
    if not os.path.exists("build/build_options.txt"):
        return False
    build_options_file = open('build/build_options.txt', 'r+')
    rawlines = build_options_file.readlines()
    lines = []
    for line in rawlines:
        lines.append(line.rstrip('\n'))
    if "training" not in lines:
        return False
    if "preprocessing" not in lines:
        return False
    return True


if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

if not check_build():
    subprocess.check_call(["./compile.py"])

if not os.path.exists("data"):
    subprocess.check_call(["./configure.sh"])
else:
    if not os.path.exists("data/positive"):
        subprocess.check_call(["mkdir", "data/positive"])
        subprocess.check_call(["./build/preprocess"])
        subprocess.check_call(["rm", "-rf", "data/lfwcrop_grey/"])


cascades = list(glob.iglob('checkpoints/*.dat'))

if len(cascades) > 0:
    most_recent_cascade = min(cascades, key=os.path.getctime)
    subprocess.check_call(["./build/train", "data/positive", "data/negative", most_recent_cascade])
else:
    subprocess.check_call(["./build/train", "data/positive", "data/negative"])
