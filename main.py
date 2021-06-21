import matplotlib.pyplot as plt
import sys
import cv2 as cv
import os
import numpy as np
from PIL import Image

def loadOutputs():
    answers = np.array([[0, 0, 0]])

    photos = []
    photos_names = []
    
    paths = ["like","peace","palm"]

    for i in range(24):
        for j in range(3):
            answer = [0 for k in range(3)]

            photos_names.append(os.listdir(paths[j])[i])

            image = cv.imread(paths[j] + "/" + os.listdir(paths[j])[i])
            converted = convertImage(image)
            photos.append(converted)

            answer[j] = 1
            answer = np.array([answer])
            answers = np.concatenate((answers, answer), axis=0)

    answers = np.delete(answers, 0, 0)
    return photos_names, photos, answers


def softmax_numpy(scores):
    return np.exp(scores) / sum(np.exp(scores))


def softmax_derivative(score):
    lin_score = score.reshape(-1)
    x = lin_score[0]
    y = lin_score[1]
    z = lin_score[2]

    soft_sum = (np.exp(x) + np.exp(y) + np.exp(z)) ** 2

    x_1 = (np.exp(x * y) + np.exp(x * z)) / soft_sum
    y_1 = (np.exp(y * x) + np.exp(y * z)) / soft_sum
    z_1 = (np.exp(z * x) + np.exp(z * y)) / soft_sum

    return np.array([x_1, y_1, z_1]).reshape(1, 3)


def convertImage(image):
    height = 32
    width = 32

    img_rgb = cv.resize(image, (height, width))
    central_color = img_rgb[15, 15]

    R = central_color[0]
    G = central_color[1]
    B = central_color[2]

    R_diff = 60
    G_diff = 60
    B_diff = 60

    for i in range(height):
        for j in range(width):
            pixel = img_rgb[i, j]
            if (R - R_diff <= pixel[0] <= R + R_diff and G - G_diff <= pixel[1] <= G + G_diff and B - B_diff <= pixel[
                2] <= B + B_diff):
                img_rgb[i, j] = [255, 255, 255]
            else:
                img_rgb[i, j] = [0, 0, 0]

    grayscale = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

    return grayscale


def createFilters(n, size):
    filters = []

    for i in range(n):
        filters.append(createMask(size))

    return filters


def createMask(size):
    return (np.array((np.random.rand(size, size) * 2)) - 1)*0.3

def generateWeights(N, M):
    return np.array((2 * np.random.random((N, M)) - 1)) * 0.1


def generateBiases(size):
    return np.array((2 * np.random.random((size, 1)) - 1)) * 0.01


def reLU(mat):
    return np.minimum(np.maximum(0, mat), 255)


def reLU_derivative(mat):
    mat[mat <= 0] = 0
    mat[mat > 0] = 1
    return mat


def tanh_derivative(mat):
    return 1 - np.tanh(mat)**2

def maxPoolingGray(image, stride):
    new_image = np.zeros([int(image.shape[0] / 2), int(image.shape[1] / 2)], dtype=np.float)

    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j] = np.amax(image[i * stride:i * stride + stride, j * stride:j * stride + stride])

    return new_image


def convolutionGray(image, mask):
    bd = mask.shape[0]
    mat2 = np.zeros([image.shape[0] - bd+1, image.shape[1] - bd+1], dtype=np.float)

    for i in range(0, image.shape[0] - bd+1):
        for j in range(0, image.shape[1] - bd+1):
            mat2[i, j] = np.sum(
                np.multiply(mask, image[i:i + bd, j:j + bd]))

    return mat2


def main():
    learning_rate = 0.003

    file_names, photos, training_outputs = loadOutputs()

    epochs = 10
    results = []

    for fold in range(8):
      # Filters Initialization
      filters_1 = createFilters(8, 5)
      filters_2 = createFilters(4, 5)

      # Weights Initialization
      weights_1 = generateWeights(5*5*32, 5*5*32)
      weights_2 = generateWeights(5*5*32, 3)

      # Biases Initialization
      biases_1 = generateBiases(5*5*32)
      biases_2 = generateBiases(3)

      print("Próba nr ", fold)
      for i in range(epochs):
          
          for k in range(len(photos)):

              if k>=fold*9 and k<(fold+1)*9:
                continue

              training_output = training_outputs[k]

              # Convolution 1

              layer_conv_1 = []

              for filter in filters_1:
                  layer_conv_1.append(convolutionGray(photos[k], filter))

              # Activation
              layer_conv_1_act = []

              for image in layer_conv_1:
                  layer_conv_1_act.append(reLU(image))

              # Maxpooling 1
              maxpooled_1 = []

              for j in range(len(layer_conv_1)):
                  maxpooled_1.append(maxPoolingGray(layer_conv_1_act[j], 2))

              # Convolution 2
              layer_conv_2 = []

              for filter in filters_2:
                  for pooled in maxpooled_1:
                      layer_conv_2.append(convolutionGray(pooled, filter))

              layer_conv_2_act = []

              for image in layer_conv_2:
                  layer_conv_2_act.append(reLU(image))

              # Maxpooling 2
              maxpooled_2 = []

              for j in range(len(layer_conv_2)):
                  maxpooled_2.append(maxPoolingGray(layer_conv_2_act[j], 2))

              # Flatten
              flattened = np.ndarray.flatten(np.concatenate(maxpooled_2))
              training_inputs = np.expand_dims(flattened, axis=0)

              # Fully Connected 1
              fully_connected_1 = training_inputs.dot(weights_1)
              fully_connected_1 = np.add(fully_connected_1, biases_1.T)
              fully_connected_1 = np.array(np.tanh(fully_connected_1))
    
              # Output
              output = np.dot(fully_connected_1, weights_2)
              output = output + biases_2.T
              output = softmax_numpy(np.reshape(output, -1))

              # Calculate error (Mean Square Error)
              cost = np.square(output - training_output).sum() / 2

              # Output Layer Adjustments
              error = output- training_output

              adjustments_2 = error * softmax_derivative(np.expand_dims(output, axis=0))

              # Fully Connected Layer Adjustments
              adjustments = np.dot(adjustments_2, weights_2.T) * tanh_derivative(fully_connected_1)
              
              adj = np.dot(adjustments, weights_1.T)

              # Filters Adjustments
              for j in range(32):
                # Filters Adjustments 2
                flat_adj = np.asarray(adj.T)

                particular_adjustments =flat_adj[j*25:(j+1)*25]

                grad_1_part_1 = np.reshape(particular_adjustments, (5, 5))
                grad_1_mask = np.equal(layer_conv_2_act[j], maxpooled_2[j].repeat(2, axis=0).repeat(2, axis=1)).astype(int)
                grad_1_window = grad_1_mask * grad_1_part_1.repeat(2, axis=0).repeat(2, axis=1)

                grad_1_part_2 = reLU_derivative(layer_conv_2[j])
                grad_1_part_3 = maxpooled_1[j//4]

                grad_1 = np.rot90(convolutionGray(grad_1_part_3, np.rot90(grad_1_window * grad_1_part_2, 2)), 2)

                # Filters Adjustments 1
                dx = np.zeros((18, 18))
                nb = dx.shape[0]
                na = (grad_1_window * grad_1_part_2).shape[0]

                lower = (nb) // 2 - (na // 2)
                upper = (nb // 2) + (na // 2)

                dx[lower:upper, lower:upper] = np.rot90(grad_1_window * grad_1_part_2, 2)

                dx = convolutionGray(dx, filters_2[j//8])

                grad_0_part_1 = dx
                grad_0_mask = np.equal(layer_conv_1_act[j//4], maxpooled_1[j//4].repeat(2, axis=0).repeat(2, axis=1)).astype(int)
                grad_0_window = grad_0_mask * grad_0_part_1.repeat(2, axis=0).repeat(2, axis=1)
                grad_0_part_2 = reLU_derivative(layer_conv_1[j//4])
                grad_0_part_3 = photos[k]

                grad_0 = np.rot90(convolutionGray(grad_0_part_3, np.rot90(grad_0_window * grad_0_part_2, 2)), 2)

                # Kernel/filters update
                filters_2[j//8] -= learning_rate/5.0 * grad_1
                filters_1[j//4] -= learning_rate/5.0 * grad_0

              # Updates
              weights_2 -= learning_rate * np.dot(fully_connected_1.T, adjustments_2)
              biases_2 -= learning_rate * np.sum(adjustments_2)

              weights_1 -= learning_rate * np.dot(training_inputs.T, adjustments)
              biases_1 -= learning_rate * np.sum(adjustments)

          # End of epoch
          print("Epoch: " + str(i))

      # Test
      well_classified = 0

      for k in range(len(photos)):
          if k>=fold*9 and k<(fold+1)*9:
            # Convolution 1
            training_output = training_outputs[k]

            layer_conv_1 = []

            for filter in filters_1:
                layer_conv_1.append(convolutionGray(photos[k], filter))

            # Activation 1
            layer_conv_1_act = []

            for image in layer_conv_1:
                layer_conv_1_act.append(reLU(image))

            # Maxpooling 1
            maxpooled_1 = []

            for j in range(len(layer_conv_1)):
                maxpooled_1.append(maxPoolingGray(layer_conv_1_act[j], 2))

            # Convolution 2
            layer_conv_2 = []

            for filter in filters_2:
                for pooled in maxpooled_1:
                    layer_conv_2.append(convolutionGray(pooled, filter))

            # Activation 2
            layer_conv_2_act = []

            for image in layer_conv_2:
                layer_conv_2_act.append(reLU(image))

            # Maxpooling 2
            maxpooled_2 = []

            for j in range(len(layer_conv_2)):
                maxpooled_2.append(maxPoolingGray(layer_conv_2_act[j], 2))

            # Flatten
            flattened = np.ndarray.flatten(np.concatenate(maxpooled_2))
            training_inputs = np.expand_dims(flattened, axis=0)

            # Fully Connected 1
            fully_connected_1 = training_inputs.dot(weights_1)
            fully_connected_1 = np.add(fully_connected_1, biases_1.T)
            fully_connected_1 = np.array(np.tanh(fully_connected_1))

            # Output
            output = np.dot(fully_connected_1, weights_2)
            output = np.add(output, biases_2.T)
            output = softmax_numpy(np.reshape(output, -1))
              
            #Accuracy History
            a = np.copy(output)
            a[a < max(a)] = 0
            a[a == max(a)] = 1

            if (a==training_output).all():
              well_classified = well_classified +1
      
      results.append(well_classified)

    for i in range(len(results)):
      print("Próba ", i, results[i],"/9")

    print("Trafność ", np.sum(np.array(results))/72.0, )




if __name__ == "__main__":
    main()
