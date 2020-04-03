import cv2 as cv

def main():
    image = cv.imread("Photos/Hand_0000002.jpg")

    img_ycbcr = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    Cb = img_ycbcr[1]
    Cr = img_ycbcr[2]

    central_color = img_ycbcr[800,600]
    print(central_color)
    Cb_color = central_color[1]
    Cr_color = central_color[2]

    Cb_diff = 15
    Cr_diff = 15

   # rows,cols = img_ycbcr.shape
    for i in range(1200):
        for j in range(1600):
            pixel = img_ycbcr[i,j]
            if(Cb_color - Cb_diff<=pixel[1]<=Cb_color+Cb_diff and Cr_color - Cr_diff<=pixel[2]<=Cr_color+Cr_diff):
                img_ycbcr[i,j]=255
            else:
                img_ycbcr[i,j]=0

    cv.namedWindow("Reka", cv.WINDOW_NORMAL)
    cv.resizeWindow("Reka", 800, 600)
    cv.imshow("Reka", img_ycbcr)

    cv.waitKey(4000)


if __name__ == "__main__":
    main()
