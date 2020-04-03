import sys

import cv2 as cv
import os
import numpy as np
from PIL import Image

def createMask(size):
    return (np.random.rand(size,size)*2)-1


def convolution(image, mask):

    mat2 = np.zeros((image.shape[0]-1, image.shape[1]-1, 3), dtype=np.int8)

    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            mat2[i-1,j-1][0]=np.sum(np.multiply(mask,image[i-1:i+2,j-1:j+2][0]))
            if(mat2[i-1,j-1][0]<0):
                mat2[i - 1, j - 1][0] =0
            mat2[i-1,j-1][1]=np.sum(np.multiply(mask,image[i-1:i+2,j-1:j+2][1]))
            if (mat2[i - 1, j - 1][1] < 0):
                mat2[i - 1, j - 1][1] = 0
            mat2[i-1,j-1][2]=np.sum(np.multiply(mask,image[i-1:i+2,j-1:j+2][2]))
            if (mat2[i - 1, j - 1][2] < 0):
                mat2[i - 1, j - 1][2] = 0

    np.set_printoptions(threshold=np.inf)
    print(mat2)
    cv.imshow("nowe",mat2)
    cv.waitKey(1500)

def main():

    #photos = os.listdir("Photos")
    #for photo in photos:
    image = cv.imread("Photos/"+"Hand_0000002.jpg")

    height = 100
    width =100

    img_rgb = cv.resize(image, (height,width))
    central_color = img_rgb[50,50]

    R = central_color[0]
    G = central_color[1]
    B = central_color[2]

    R_diff = 60
    G_diff = 60
    B_diff = 60

    for i in range(height):
        for j in range(width):
            pixel = img_rgb[i,j]
            if(R - R_diff<=pixel[0]<=R+R_diff and G - G_diff<=pixel[1]<=G+G_diff and B - B_diff<=pixel[2]<=B+B_diff):
                img_rgb[i,j]=[255,255,255]
            else:
                img_rgb[i,j]=[0,0,0]

    # przerobione

    for i in range(20):
        mask = createMask(3)
        convolution(img_rgb, mask)

    cv.namedWindow("Reka", cv.WINDOW_NORMAL)
    cv.resizeWindow("Reka", 800, 600)
    cv.imshow("Reka", img_rgb)
    cv.imwrite("save.jpg",img_rgb)
    cv.waitKey(2500)


if __name__ == "__main__":
    main()
