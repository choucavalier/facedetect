#!/usr/bin/env python3

'''Visualize mblbp features in a given window with an animation'''

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

    img = Image.open('gfx/tgy.jpg').convert('LA')

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

    rectangles = []

    for i in range(9):
        rectangle = patches.Rectangle((0, 0), 0, 0,
                                      alpha=0 if i == 4 else 0.6,
                                      linewidth=0)
        ax.add_patch(rectangle)
        rectangles.append(rectangle)

    features = []

    for block_w in range(3, window_w, 3):
        for block_h in range(3, window_h, 3):
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

    def animate(i, features, rectangles):

        color = random.choice(COLORS)

        feature = features[i]

        small_block_w = feature['block_w'] / 3
        small_block_h = feature['block_h'] / 3

        for i, rectangle in enumerate(rectangles):

            rectangle.set_x(feature['offset_x'] + (i % 3) * small_block_w)
            rectangle.set_y(feature['offset_y'] + (i // 3) * small_block_h)
            rectangle.set_width(small_block_w)
            rectangle.set_height(small_block_h)
            rectangle.set_facecolor(color)

        return rectangles

    anim = animation.FuncAnimation(fig, animate, frames=120, interval=100,
                                   blit=True, fargs=(features, rectangles))

    anim.save('gfx/animated_mblbp.mp4', fps=6, bitrate=-1,
              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'rgb24'],
              metadata={'frameon': False, 'pad_inches': 0.0,
                        'bbox_inches': 'tight', 'dpi': dpi})

    # plt.show()

def main():

    window_w = 20
    window_h = 20

    plot_animated_mblbp(window_w, window_h)

if __name__ == '__main__':
    main()
