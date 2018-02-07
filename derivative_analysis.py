import cv2
import numpy as np

import greyscale_img
import segmentation_analysis


def x_derivative(img, threshold=65, c_threshold=100, greyscale_img=True):
    matrix = np.full((img.shape[0],img.shape[1]), 255)
    f = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if greyscale_img:
                m = img[i-1:i+2,j-1:j+2]
                v = np.sum(m*f)
                if v >= threshold:
                    matrix[i,j] = 0
                else:
                    matrix[i,j] = 255 - v
            else:
                r = img[i-1:i+2,j-1:j+2,0]
                g = img[i-1:i+2,j-1:j+2,1]
                b = img[i-1:i+2,j-1:j+2,2]
                v = (np.sum(r*f)**2 + np.sum(g*f)**2 + np.sum(b*f)**2)**0.5
                if v >= c_threshold:
                    matrix[i,j] = 0
                else:
                    matrix[i,j] = 255 - v

    return matrix

def y_derivative(img, threshold=65, c_threshold=100, greyscale_img=True):
    matrix = np.full((img.shape[0],img.shape[1]), 255)
    f = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if greyscale_img:
                m = img[i-1:i+2,j-1:j+2]
                v = np.sum(m*f)
                if v >= threshold:
                    matrix[i,j] = 0
                else:
                    matrix[i,j] = 255 - v
            else:
                r = img[i-1:i+2,j-1:j+2,0]
                g = img[i-1:i+2,j-1:j+2,1]
                b = img[i-1:i+2,j-1:j+2,2]
                v = (np.sum(r*f)**2 + np.sum(g*f)**2 + np.sum(b*f)**2)**0.5
                if v >= c_threshold:
                    matrix[i,j] = 0
                else:
                    matrix[i,j] = 255 - v


    return matrix

def combine_deriv(img1, img2):
    matrix = np.zeros(img1.shape)

    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            matrix[i,j] = min(img1[i,j], img2[i,j])

    return matrix

# img = xy_derivative image
def slope_descriptor(img, w_dim=4):
    matrix = np.zeros(img.shape[0]/w_dim+1, img.shape[1]/w_dim+1)

    for i in range(0, img.shape[0]/w_dim):
        row = i * w_dim
        for j in range(0, img.shape[1]/w_dim):
            col = j * w_dim
            n, sum_xy, sum_x, sum_y, sum_x_sqr = 0, 0, 0, 0, 0
            for m in range(row, min(row+w_dim, img.shape[0])):
                for n in range(col, min(col+w_dim, img.shape[1])):
                    if img[m,n] == 0:
                        n = n+1
                        sum_xy = sum_xy + m*n
                        sum_x = sum_x + n
                        sum_y = sum_y + m
                        sum_x_sqr = sum_x_sqr + n*n

            matrix[i,j] = -1
            if n > 1 and n <= w_dim * w_dim:
                matrix[i,j] = (n*sum_xy - sum_x*sum_y) / (n*sum_x_sqr - sum_x*sum_x)  #slope formula - linear regression

    return matrix

def gaussian_smoother(img):
    matrix = np.full(img.shape, 255)
    f = np.array([[1,4,7,4,1], [4,16,26,16,4], [7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]])

    for i in range(2, img.shape[0] - 2):
        for j in range(2, img.shape[1] - 2):
            m = img[i-2:i+3,j-2:j+3]
            matrix[i,j] = np.sum(m*f) / 273

    return matrix

def xy_deriv_segments(xy_deriv, segments):
    matrix = np.full(xy_deriv.shape, 255)
    marked_segs = {}

    for i in range(0, xy_deriv.shape[0]):
        for j in range(0, xy_deriv.shape[1]):
            if xy_deriv[i,j] == 0:
                marked_segs[segments[i,j]] = 1

    for i in range(0, xy_deriv.shape[0]):
        for j in range(0, xy_deriv.shape[1]):
            if segments[i,j] in marked_segs:
                matrix[i,j] = 0

    return matrix

def display(img, descriptor):
    cv2.imshow(descriptor, img.astype(np.uint8))
    cv2.waitKey(0)

def derivative_analysis():
    img = cv2.imread("./images/birdo.jpg", cv2.IMREAD_UNCHANGED)
    grey_img = greyscale_img.generate(img)
    gauss_img = gaussian_smoother(grey_img)
    segments = segmentation_analysis.run_slic(grey_img)

    x_deriv = x_derivative(gauss_img)
    y_deriv = y_derivative(gauss_img)
    xy_deriv = combine_deriv(x_deriv, y_deriv)
    border_segments = xy_deriv_segments(xy_deriv, segments)

    #x_deriv_color = x_derivative(img, greyscale_img=False)
    #y_deriv_color = y_derivative(img, greyscale_img=False)
    #xy_deriv_color = combine_deriv(x_deriv_color, y_deriv_color)


    #cv2.imwrite("./outputs/birdo_gauss.png", gauss_img)
    #cv2.imwrite("./outputs/birdo_x_deriv.png", x_deriv)
    #cv2.imwrite("./outputs/birdo_y_deriv.png", y_deriv)

    #cv2.imwrite("./outputs/birdo_xy_deriv_color.png", xy_deriv_color)
    cv2.imwrite("./outputs/birdo_xy_deriv.png", xy_deriv)
    cv2.imwrite("./outputs/birdo_xy_deriv_segments.png", border_segments)


if __name__ == '__main__':
    derivative_analysis()
