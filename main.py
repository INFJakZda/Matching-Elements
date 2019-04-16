import matplotlib.pyplot as plt
import sys
import os
import re
from itertools import product
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
        # results of pairs of objects
        self.scores_pair = {}
        # images normally and reversed { img1: [up, down], ...}
        self.images_up_and_down = {}
        # const for recognize images in images_up_and_down dict 
        self.IMG_UP = "img_up"
        self.IMG_DOWN = "img_down"

        self.pool = ThreadPool(no_threads)
        
    def process_images(self):
        for image_no in range(self.N):
            # indexes of all images without image_no
            images_indexes = np.delete(np.arange(self.N), image_no)
            # pairs - image_no with all others
            pairs = list(product([image_no], images_indexes))
            # get ranking
            ranking = np.array(self.pairs_ranking(pairs))
            print_ranking = list(map(lambda x: str(x[1]), ranking[:,0]))
            print(' '.join(print_ranking))

    def pairs_ranking(self, pairs):
        # multiprocessing result comparision
        results = self.pool.map(self.result_comparison, pairs)
        # return sorted tuples ((pair), result)
        return sorted(results, key = lambda x: x[1])

    def result_comparison(self, pair):
        if (pair[1], pair[0]) in self.scores_pair:
            return (pair, self.scores_pair[(pair[1], pair[0])])
        # get images path
        img1_path = os.path.join(self.data_path, str(pair[0]) + ".png")
        img2_path = os.path.join(self.data_path, str(pair[1]) + ".png")

        img1 = self.prepare_image(img1_path)[self.IMG_UP]
        img2 = self.prepare_image(img2_path)[self.IMG_DOWN]

        score = self.result_calculation(img1, img2)
        self.scores_pair[pair] = score
        return (pair, score)

    def prepare_image(self, img_path):
        # if there is calculated image return it
        if img_path in self.images_up_and_down:
            return self.images_up_and_down[img_path]

        # read image
        img = io.imread(img_path)
        # find a basis of the image and set image to it
        img = self.crop_to_object(img)
        # rotate image for feature comparision
        img_up, img_down = self.rotate_basis(img)
        # save the results for feature use
        self.images_up_and_down[img_path] = {
            self.IMG_UP: self.get_contours(img_up), 
            self.IMG_DOWN: self.get_contours(img_down)
        }
        return self.images_up_and_down[img_path]
           
    
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
