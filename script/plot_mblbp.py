#!/usr/bin/env python3

'''Visualize randomly chosen mblbp features in a given window'''

import random
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.image as mpimg
from PIL import Image

COLORS = ['#1a535c', '#4ecdc4', '#ff6b6b', '#ffe66d', '#ffe66d',
          '#ff6b6b', '#4ecdc4', '#1a535c']

def plot_animated_mblbp(window_w, window_h):

    dpi = 96

    img = Image.open('gfx/prisca.jpg').convert('LA')

    fig = plt.figure(figsize=(dpi / 40, dpi / 40), dpi=dpi, frameon=False)

    ax = plt.axes(xlim=(0, 20), ylim=(0, 20))
    ax.imshow(img, interpolation='none', cmap=plt.get_cmap('gray'),
              extent=[0, 20, 20, 0], alpha=0.7)
    ax.set_ylim(ax.get_ylim()[::-1]) # invert y-axis
    ax.xaxis.tick_top() # move x-axis to the top
    ax.xaxis.set_ticks(range(1, 21))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticks(range(1, 21))
    ax.yaxis.set_ticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    ax.grid(True, which='both', linestyle='-')
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    features = []

    for block_w in range(3, 10, 3):
        for block_h in range(3, 10, 3):
            for offset_x in range(window_w - block_w + 1):
                for offset_y in range(window_h - block_h + 1):
                    feature = {
                        'block_w': block_w,
                        'block_h': block_h,
                        'offset_x': offset_x,
                        'offset_y': offset_y,
                    }
                    features.append(feature)

    random.shuffle(features)
    sample = random.sample(features, 6)

    for i, feature in enumerate(sample):
        rectangle = patches.Rectangle(
            (feature['offset_x'], feature['offset_y']),
            feature['block_w'], feature['block_h'],
            facecolor=COLORS[i], linewidth=1, alpha=0.4,
        )
        ax.add_patch(rectangle)

    plt.show()

def main():

    window_w = 20
    window_h = 20

    plot_animated_mblbp(window_w, window_h)

if __name__ == '__main__':
    main()
