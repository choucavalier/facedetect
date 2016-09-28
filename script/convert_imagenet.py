import os
from PIL import Image
from random import randint

IMAGENET_DIR = 'data/val/'
NONFACES_DIR = 'data/non_faces/'

IMAGENET_PREFIX = "ILSVRC2010_val_"
NONFACES_PREFIX = "NONFACE_"

INDEX_NUMBER_LENGTH = 8

IMAGE_SIZE = 20
SUBDIVISIONS_PER_IMAGE = 3
MAX_SUBDIVISION_SCALE_FACTOR = 0.5

def main():

    if not os.path.exists(IMAGENET_DIR):
        print("ERROR: IMAGENET DIRECTORY DOES NOT EXIST")
    if not os.path.exists(NONFACES_DIR):
        os.makedirs(NONFACES_DIR)

    file_list = []
    for root, dirs, files in os.walk(IMAGENET_DIR):
        for f in files:
            file_list.append(root + f)

    file_list_len = len(file_list)
    i = 0
    file_list = sorted(file_list)
    for f in file_list:
        progress = int( ( ( float(i)/float(SUBDIVISIONS_PER_IMAGE) ) / float(file_list_len) ) * 100.0 )
        print(f + "    progress : " + int_to_normalised_string(progress, 2) + " %")
        jpgfile = Image.open(f)
        jpgfile = jpgfile.convert('L')
        subdivisions = get_random_subdivisions(jpgfile.getbbox(), SUBDIVISIONS_PER_IMAGE)

        for subdivision in subdivisions:
            cropped_image = jpgfile.crop(subdivision)
            cropped_image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            cropped_image.save(NONFACES_DIR + NONFACES_PREFIX + int_to_normalised_string(i, INDEX_NUMBER_LENGTH) + ".png", "PNG")
            i += 1

'''
Returns list of tuples of (x1, y1, x2, y2) subdivisions
of image bounding box

Image bounding box will be the non-zero bounding box of
the image
'''
def get_random_subdivisions(bbox, n_subdivisions):

    bbox_x1 = bbox[0]
    bbox_y1 = bbox[1]
    bbox_x2 = bbox[2]
    bbox_y2 = bbox[3]

    img_width = bbox_x2 - bbox_x1
    img_height = bbox_y2 - bbox_y1

    subdivisions = []

    for i in range(0, n_subdivisions):

        size = randint(IMAGE_SIZE, int(min(img_width*MAX_SUBDIVISION_SCALE_FACTOR, img_height*MAX_SUBDIVISION_SCALE_FACTOR)))

        x1 = randint(bbox_x1, bbox_x2 - size - 1)
        y1 = randint(bbox_y1, bbox_y2 - size - 1)
        x2 = x1 + size
        y2 = y1 + size

        subdivisions.append([x1, y1, x2, y2])

    return subdivisions


def int_to_normalised_string(i, n):

    if i == 0:
        string = "0" * n
    else:
        string = ""
        p = 0
        k = 1
        while k <= i:
            k *= 10
            p += 1
        for j in range(0, n - p):
            string += "0"
        string += str(i)
    return string



if __name__ == '__main__':
    main()
