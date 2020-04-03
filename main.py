import cv2 as cv


def main():
    image = cv.imread("Photos/Hand_0000002.jpg", 0)
    # image = cv.cvtColor(image,)

    thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(image, 127, 255, 0)

    cv.namedWindow("Reka", cv.WINDOW_NORMAL)
    cv.resizeWindow("Reka", 800, 600)
    cv.imshow("Reka", thresh)

    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    '''
    edged = cv.Canny(image, 25, 160)
    contours, hierarchy = cv.findContours(edged,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    cv.namedWindow("Reka",cv.WINDOW_NORMAL)
    cv.resizeWindow("Reka",800,600)
    cv.imshow("Reka",image)
    '''

    cv.waitKey(4000)


if __name__ == "__main__":
    main()
