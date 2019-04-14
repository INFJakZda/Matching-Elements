import matplotlib.pyplot as plt
import sys
import os
import re
import itertools
from multiprocessing.dummy import Pool as ThreadPool 

import cv2
import numpy as np

from skimage import io
from skimage.transform import rotate, resize, rescale
from skimage.util import img_as_bool, img_as_float, pad, invert, img_as_ubyte
from skimage.morphology import square, closing, dilation
from skimage.color import gray2rgb, rgb2gray

class ImageClassifier:
    def __init__(self, N, data_path, no_threads):
        self.N = N
        self.data_path = data_path
        self.images_path = list(map(lambda n: os.path.join(self.data_path, str(n) + ".png"), np.arange(0, self.N)))
        self.pair_scores = {}       #TODO
        self.imgs_rotations = {}    #TODO
        self.pool = ThreadPool(no_threads)   #TODO

    def process_images(self):
        pass

    
if __name__ == "__main__":
    # number of threads on your computer
    no_threads = 8

    # error - bad number of args
    if len(sys.argv) != 3:
        print(os.path.basename(sys.argv[0]) + " <path> <N>")
        exit(1)

    # read path & N - number of pictures
    data_path = os.path.abspath(sys.argv[1])
    N = int(sys.argv[2])
    
    # create & run classifier
    classifier = ImageClassifier(N, data_path, no_threads)
    classifier.process_images()
