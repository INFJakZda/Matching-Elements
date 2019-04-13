import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np

from skimage import io
from skimage.transform import rotate, resize, rescale
from skimage.util import img_as_bool, img_as_float, pad, invert, img_as_ubyte
from skimage.morphology import square, closing, dilation
from skimage.color import gray2rgb, rgb2gray

def getBottomLine(image):
    edges = cv2.Canny(image, 50, 150, apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    count =0
    for rho,theta in lines[0]:
        # print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if( 20 < 180*theta/np.pi < 88):
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if (160 > 180 * theta / np.pi > 93):
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imwrite('detectP.jpg',image)
    io.imshow(image)
    io.show()

def get_contours(img):
    img = np.pad(img,(5,5),'constant', constant_values=(0, 0))
    img = img_as_ubyte(closing(img, square(7)))
    ret,thresh = cv2.threshold(img,10,255,0)
    _, img_c = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(img_c) > 1:
        # print("more than one contour")
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

def process_image(n, N, data_path):
    filename = os.path.join(data_path, str(n) + ".png")
    image = io.imread(filename, as_gray=True)
    # image = io.imread(filename)

    getBottomLine(image)

    # contours = get_contours(image)
    # img = rotate_basis(image)

    # io.imshow(img)
    # io.show()

    

    

    # print(image)
    # io.imshow(image)
    # io.show()
    # exit(1)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(os.path.basename(sys.argv[0]) + " <path> <N>")
        exit(1)

    data_path = os.path.abspath(sys.argv[1])
    N = int(sys.argv[2])
    
    for n in range(N):
        process_image(n, N, data_path)
        print("3 2")
