import cv2
import numpy as np

import greyscale_img


def x_derivative(img, threshold=65):
    matrix = np.full(img.shape, 255)
    f = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            m = img[i-1:i+2,j-1:j+2]
            v = np.sum(m*f)
            if v >= threshold:
                matrix[i,j] = 0
            else:
                matrix[i,j] = 255 - v

    return matrix

def y_derivative(img, threshold=65):
    matrix = np.full(img.shape, 255)
    f = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            m = img[i-1:i+2,j-1:j+2]
            v = np.sum(m*f)
            if v >= threshold:
                matrix[i,j] = v
            else:
                matrix[i,j] = 255 - v

    return matrix

def combine_deriv(img1, img2):
    matrix = np.zeros(img1.shape)

    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            matrix[i,j] = min(img1[i,j], img2[i,j])

    return matrix

def gaussian_smoother(img):
    matrix = np.full(img.shape, 255)
    f = np.array([[1,4,7,4,1], [4,16,26,16,4], [7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]])

    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            m = img[i-2:i+3,j-2:j+3]
            matrix[i,j] = np.sum(m*f) / 273

    return matrix

def display(img, descriptor):
    cv2.imshow(descriptor, img.astype(np.uint8))
    cv2.waitKey(0)

def derivative_analysis():
    img = cv2.imread("./images/birdo.jpg", cv2.IMREAD_UNCHANGED)
    grey_img = greyscale_img.generate(img)
    gauss_img = gaussian_smoother(grey_img)
    x_deriv = x_derivative(gauss_img)
    y_deriv = y_derivative(gauss_img)
    xy_deriv = combine_deriv(x_deriv, y_deriv)

    cv2.imwrite("./outputs/birdo_gauss.png", gauss_img)
    cv2.imwrite("./outputs/birdo_x_deriv.png", x_deriv)
    cv2.imwrite("./outputs/birdo_y_deriv.png", y_deriv)
    cv2.imwrite("./outputs/birdo_xy_deriv.png", xy_deriv)


if __name__ == '__main__':
    derivative_analysis()
