import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate, resize, rescale
from skimage.util import img_as_bool, img_as_float, pad, invert, img_as_ubyte
from skimage.morphology import square, closing, dilation
from skimage.color import gray2rgb, rgb2gray
import time
from multiprocessing.dummy import Pool as ThreadPool 
import sys
import re
import itertools


def crop_to_object(img):
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

def get_contours(img):
    
    img = pad(img,(5,5),'constant', constant_values=(0, 0))
    img = img_as_ubyte(closing(img, square(7)))
    ret,thresh = cv2.threshold(img,10,255,0)
    _, img_c, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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


def rotate_basis(img): #return ok ud
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

    
def calc_shape_context_distance(img_1, img_2):
    sd = cv2.createShapeContextDistanceExtractor()
    try:
        d2 = sd.computeDistance(img_1,img_2)
        return d2
    except Exception as e:
        print(e)
        return 100


    
    
def get_img_contours(img_path):
    if img_path not in imgs_rotations:
        img = io.imread(img_path)
        img = crop_to_object(img)
        ok, ud = rotate_basis(img)
        imgs_rotations[img_path] = {"ok":get_contours(ok), "ud":get_contours(ud)}
    return imgs_rotations[img_path]
    
def score_pair(img1_path, img2_path):
    if (img2_path,img1_path) in pair_scores:
        return pair_scores[(img2_path,img1_path)]
    start = time.time()
    l = get_img_contours(img1_path)["ok"]
    r = get_img_contours(img2_path)["ud"]
    score = calc_shape_context_distance(l,r) * -1
    pair_scores[(img1_path,img2_path)] = score
    return score


def score_similarity(pair):
    return (pair,score_pair(pair[0],pair[1]))

def rank_pairs(pairs):
    scores = pool.map(score_similarity, pairs)
    return sorted(scores, key=lambda x: -x[1])

pair_scores = {}
imgs_rotations = {}
imgs_objects = {}
pool = ThreadPool(8) 

path = sys.argv[1]
N = sys.argv[2]
img_paths = list(map(lambda x: path + "/" + str(x) + ".png", np.arange(0,int(N))))
for img_path in img_paths:
    candidates = set(img_paths)
    candidates.remove(img_path)
    possible_pairs = list(itertools.product([img_path],candidates))
    rank = np.array(rank_pairs(possible_pairs))
    rank_f = list(map(lambda x: re.search('(\d+).png',x[1]).group(1), rank[:,0]))
    print(' '.join(rank_f))
