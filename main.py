import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np

from skimage import io

def process_image(n, N, data_path):
    filename = os.path.join(data_path, str(n) + ".png")
    image = io.imread(filename, as_gray=True)
    
    # for px in image:
    #     print(px)

    edges = cv2.Canny(image, 50, 150, apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    count =0
    for rho,theta in lines[0]:
        print(rho, theta)
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

    cv2.imwrite('detectP.jpg',image)
    io.imshow(image)
    io.show()

    

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
