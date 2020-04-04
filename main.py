import sys

import cv2 as cv
import os
import numpy as np
from PIL import Image


def createMask(size):
    return (np.random.rand(size,size)*0.000002)-0.000001


def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))


def sigmoid_derivative(x):
    return x * (1 - x)


def maxPoolingGray(image, stride):
    new_image = np.zeros([int(image.shape[0]/2),int(image.shape[1]/2)], dtype=np.float)

    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j] = np.amax(image[i * stride:i * stride + stride, j * stride:j * stride + stride])

    return new_image


def maxPooling(image, stride):

    new_image = np.zeros((int(image.shape[0]/2), int(image.shape[1]/2), 3), dtype=np.int8)
    '''
    #mat = image[0:2, 0:2][0]
    mat = image[0,0]
    print(mat)
    image[0,1] = [1,2,3]
    mat = image[0,1]
    print(mat)

    #mat = image[0:2, 0:2]
    # mat[0,0]=1
    print(mat)

    mat = max(image[0:2, 0:2].reshape(4,3)[:,0])
    #mat[0,0]=1
    print(mat)
    '''
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i,j][0] = max(image[i*stride:i*stride+stride,j*stride:j*stride+stride].reshape(4,3)[:,0])
            new_image[i,j][1] = max(image[i*stride:i*stride+stride,j*stride:j*stride+stride].reshape(4,3)[:,1])
            new_image[i,j][2] = max(image[i*stride:i*stride+stride,j*stride:j*stride+stride].reshape(4,3)[:,2])

    return new_image


def convolutionGray(image, mask):
    mat2 = np.zeros([image.shape[0] - 1, image.shape[1] - 1], dtype=np.float)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            mat2[i - 1, j - 1] = np.sum(
                np.multiply(mask, image[i - 1:i + 2, j - 1:j + 2]))
            # reLU
            mat2[i - 1, j - 1] = max(0, mat2[i - 1, j - 1])

    return mat2


def convolution(image, mask):

    mat2 = np.zeros((image.shape[0]-1, image.shape[1]-1, 3), dtype=np.int8)

    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            mat2[i - 1, j - 1][0] = np.sum(np.multiply(mask, image[i - 1:i + 2, j - 1:j + 2].reshape(9, 3)[:, 0].reshape(3,3)))
            # reLU RED
            mat2[i-1,j-1][0]=max(0,mat2[i-1,j-1][0])

            mat2[i-1,j-1][1]=np.sum(np.multiply(mask,image[i - 1:i + 2, j - 1:j + 2].reshape(9, 3)[:, 1].reshape(3,3)))
            # reLU GREEN
            mat2[i - 1, j - 1][1] = max(0, mat2[i - 1, j - 1][1])

            mat2[i-1,j-1][2]=np.sum(np.multiply(mask,image[i - 1:i + 2, j - 1:j + 2].reshape(9, 3)[:, 2].reshape(3,3)))
            # reLU BLUE
            mat2[i - 1, j - 1][2] = max(0, mat2[i - 1, j - 1][2])

    return mat2

def main():

    np.random.seed(1)
    # Initialize training specifications
    training_inputs = np.empty((4,14406))
    training_outputs = np.array([[1,1,0,1]]).T
    #training_outputs = np.array([[0,0,1,0]]).T
    synaptic_weights_1 = 0.2 * np.random.random((14406, 1)) - 0.1
    #print(synaptic_weights_1)
    print(np.sum(synaptic_weights_1))
    #synaptic_weights_2 = 2 * np.random.random((14406, 1)) - 1
    filter_num_1 = 6
    filters_1 = []
    vec_num = 0

    for i in range(filter_num_1):
        filters_1.append(createMask(3))

    # Getting Data
    photos = os.listdir("Photos/like")
    for photo in photos:
        image = cv.imread("Photos/like/"+photo)

        height = 100
        width = 100

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

        grayscale = cv.cvtColor(img_rgb,cv.COLOR_RGB2GRAY)

        new_images_1 = []

        # Convolution Layer 1
        for i in range(filter_num_1):
            new_images_1.append(convolutionGray(grayscale, filters_1[i]))
        # Max Pooling Layer 1
        for i in range(filter_num_1):
            new_images_1[i] = maxPoolingGray(new_images_1[i], 2)

        # Flattening Layer
        flattened = np.ndarray.flatten(np.concatenate(new_images_1))
        training_inputs[vec_num]=flattened
        vec_num=vec_num+1

    #learning

    for iteration in range(100000):
        input_layer = training_inputs
        #print(np.array(np.dot(input_layer, synaptic_weights_1)))
        outputs = sigmoid(np.array(np.dot(input_layer, synaptic_weights_1)))

        error = training_outputs - outputs

        adjustments = error * sigmoid_derivative(outputs)

        synaptic_weights_1 += np.dot(input_layer.T, adjustments)

        if iteration%2000 == 0:
             print("Outputs")
             print(outputs)
            # print("Error")
            # print(error)
            # print("Adjustments")
            # print(adjustments)
            # print(synaptic_weights_1)

    print("Weights after training")
    print(np.sum(synaptic_weights_1))
    #print(synaptic_weights_1)

    # testing
    photos = os.listdir("Photos/victory")
    for photo in photos:
        image = cv.imread("Photos/victory/"+photo)

        height = 100
        width = 100

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

        grayscale = cv.cvtColor(img_rgb,cv.COLOR_RGB2GRAY)

        cv.imshow("zdj", grayscale)
        cv.waitKey(1500)

        new_images_1 = []

        # Convolution Layer 1
        for i in range(filter_num_1):
            new_images_1.append(convolutionGray(grayscale, filters_1[i]))
        # Max Pooling Layer 1
        for i in range(filter_num_1):
            new_images_1[i] = maxPoolingGray(new_images_1[i], 2)

        # Flattening Layer
        flattened = np.ndarray.flatten(np.concatenate(new_images_1))
        outputs = sigmoid(np.array(np.dot(flattened, synaptic_weights_1)))
        print("Test outputs")
        print(outputs)



if __name__ == "__main__":
    main()