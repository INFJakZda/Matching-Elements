import matplotlib.pyplot as plt
import sys
import os
import re
from itertools import product
# from multiprocessing.dummy import Pool as ThreadPool 

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

        # self.pool = ThreadPool(no_threads)
        
    def process_images(self):
        for image_no in range(self.N):
            # indexes of all images without image_no
            images_indexes = np.delete(np.arange(self.N), image_no)
            # pairs - image_no with all others
            pairs = list(product([image_no], images_indexes))
            #print("pairs: ", pairs)
            # get ranking
            ranking = np.array(self.pairs_ranking(pairs))
            print_ranking = list(map(lambda x: str(x[1]), ranking[:,0]))
            print(' '.join(print_ranking))

    def pairs_ranking(self, pairs):
        # multiprocessing result comparision
        results = map(self.result_comparison, pairs)
        # return sorted tuples ((pair), result)
        return sorted(results, key = lambda x: x[1])

    def result_comparison(self, pair):
        if (pair[1], pair[0]) in self.scores_pair:
            return (pair, self.scores_pair[(pair[1], pair[0])])
        # get images path
        img1_path = os.path.join(self.data_path, str(pair[0]) + ".png")
        img2_path = os.path.join(self.data_path, str(pair[1]) + ".png")

        img1 = self.prepare_image(img1_path, pair[0])[self.IMG_UP]
        img2 = self.prepare_image(img2_path, pair[1])[self.IMG_DOWN]

        score = self.result_calculation(img1, pair[0], img2, pair[1])
        self.scores_pair[pair] = score
        return (pair, score)

    def prepare_image(self, img_path, img_nr):
        # if there is calculated image return it
        if img_nr in self.images_up_and_down:
            return self.images_up_and_down[img_nr]
        # read image
        img = io.imread(img_path)
        # find a basis of the image and set image to it
        img = self.rotate_image(img)
        # cut image from rotated img
        img = self.cut_image(img)
        # rotate image for feature comparision
        img_up, img_down = self.rotate_basis(img)
        #plt.imshow(img_up)
        #plt.show()
        #plt.imshow(img_down)
        #plt.show()
        # save the results for feature use
        self.images_up_and_down[img_nr] = {
            self.IMG_UP: img_up,
            self.IMG_DOWN: img_down
        }
        return self.images_up_and_down[img_nr]

    def rotate_image(self, img):
        img = img_as_ubyte(closing(img, square(5)))
        _, thresh = cv2.threshold(img,20,255,0)
        _, lc, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(lc) == 1
        rect = cv2.minAreaRect(lc[0])
        img = rotate(img, rect[2], resize = True)
        return img
    
    def cut_image(self, img):
        img = img_as_ubyte(img)
        _, thresh = cv2.threshold(img,20,255,0)
        thresh = img_as_ubyte(closing(thresh, square(5)))
        _, lc, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(lc) == 1
        rect = cv2.minAreaRect(lc[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        min_y, max_y = min(box[:,1]), max(box[:,1])
        min_x, max_x = min(box[:,0]), max(box[:,0])
        return img[min_y:max_y,min_x:max_x]

    def rotate_basis(self, img):
        if img.shape[0]> img.shape[1]:
            img = np.rot90(img).copy()
        img = resize(img,(100,150))
        h, _ = img.shape
        upper_z = np.count_nonzero(img[0:int(h/2)])
        bottom_z = np.count_nonzero(img[int(h/2):h])
        if bottom_z < upper_z:
            return np.rot90(img, 2).copy(), img.copy()
        else:
            return img.copy(), np.rot90(img,2).copy()

    def result_calculation(self, img1, img1_nr, img2, img2_nr):
        sum_of_amplitude = 0
        last = 0
        for i in range(img1.shape[1]):
            white = 0
            for j in range(img1.shape[0]):
                if img1[j][i] != 0:
                    white += img1[j][i]
                if img2[j][i] != 0:
                    white += img2[j][i]
            if last == 0:
                last = white
            else:
                sum_of_amplitude += np.abs(last - white)
                last = white
        return sum_of_amplitude

    '''def result_calculation(self, img1, img1_nr, img2, img2_nr):
        sum_of_lengths = []
        for i in range(img1.shape[1]):
            white = 0
            for j in range(img1.shape[0]):
                if img1[j][i] > 0.01:
                    white += img1[j][i]
                if img2[j][i] > 0.01:
                    white += img2[j][i]
            sum_of_lengths.append(white)
        result = np.var(sum_of_lengths)
        return result
        '''

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
