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
        self.scores_pair = {}
        self.images_up_and_down = {}
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

        img1 = self.prepare_image(img1_path)["ok"]
        img2 = self.prepare_image(img2_path)["ud"]

        score = self.result_calculation(img1, img2)
        self.scores_pair[pair] = score
        return (pair, score)

    def prepare_image(self, img_path):
        if img_path not in self.images_up_and_down:
            img = io.imread(img_path)
            img = self.crop_to_object(img)
            ok, ud = self.rotate_basis(img)
            self.images_up_and_down[img_path] = {
                "ok": self.get_contours(ok), 
                "ud": self.get_contours(ud)
            }
        return self.images_up_and_down[img_path]

    def get_contours(self, img):
        img = pad(img,(5,5),'constant', constant_values=(0, 0))
        img = img_as_ubyte(closing(img, square(7)))
        ret, thresh = cv2.threshold(img,10,255,0)
        _, img_c, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(img_c) > 1:
            print("more than one contour")
            if len(img_c) == 2:
                if len(img_c[0]) > len(img_c[1]):
                    return img_c[0]
                else:
                    return img_c[1]
            assert len(img_c) <= 2
            #img_d = np.copy(thresh)    
            #img_d = gray2rgb(img_d)
            #cv2.drawContours(img_d, img_c, 0, (0,255,0), 3)
            #io.imshow(img_d)
        return img_c[0]

    def rotate_basis(self, img): #return ok ud
        if img.shape[0]> img.shape[1]:
            img = np.rot90(img).copy()
        img = resize(img,(100,150))
        h,w = img.shape
        upper_z = np.count_nonzero(img[0:int(h/2)])
        bottom_z = np.count_nonzero(img[int(h/2):h])
        if bottom_z < upper_z:
            return np.rot90(img, 2).copy(), img.copy()
        else:
            return img.copy(), np.rot90(img,2).copy()

    def result_calculation(self, img_1, img_2):
        sd = cv2.createShapeContextDistanceExtractor()
        try:
            d2 = sd.computeDistance(img_1,img_2)
            return d2
        except Exception as e:
            print(e)
            return 100

    def crop_to_object(self, img):
        img = img_as_ubyte(closing(img, square(5)))
        ret,thresh = cv2.threshold(img,20,255,0)
        _, lc, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        assert len(lc) == 1
        rect = cv2.minAreaRect(lc[0])
        img = rotate(img, rect[2], resize = True)
        img = img_as_ubyte(img)
        ret,thresh = cv2.threshold(img,20,255,0)
        thresh = img_as_ubyte(closing(thresh, square(5)))
        _, lc, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        assert len(lc) == 1
        rect = cv2.minAreaRect(lc[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        min_y, max_y = min(box[:,1]), max(box[:,1])
        min_x, max_x = min(box[:,0]), max(box[:,0])
        return img[min_y:max_y,min_x:max_x]

    
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
