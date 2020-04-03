import cv2 as cv
import os
import numpy as np

def main():

    photos = os.listdir("Photos")
    for photo in photos:


        image = cv.imread("Photos/"+photo)

        height = 100
        width =100

        img_ycbcr = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        Cb = img_ycbcr[1]
        Cr = img_ycbcr[2]
        img_ycbcr = cv.resize(img_ycbcr, (height,width))
        central_color = img_ycbcr[50,50]

        Cb_color = central_color[1]
        Cr_color = central_color[2]

        Cb_diff = 20
        Cr_diff = 20


       # rows,cols = img_ycbcr.shape
        for i in range(height):
            for j in range(width):
                pixel = img_ycbcr[i,j]
                if(Cb_color - Cb_diff<=pixel[1]<=Cb_color+Cb_diff and Cr_color - Cr_diff<=pixel[2]<=Cr_color+Cr_diff):
                    img_ycbcr[i,j]=255
                else:
                    img_ycbcr[i,j]=0

        cv.namedWindow("Reka", cv.WINDOW_NORMAL)
        cv.resizeWindow("Reka", 800, 600)
        cv.imshow("Reka", img_ycbcr)

        cv.waitKey(3000)


if __name__ == "__main__":
    main()
