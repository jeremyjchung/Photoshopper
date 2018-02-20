from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

from coordinate import Coordinate
import greyscale_img
import segmentation_analysis


def x_derivative(img, threshold=65, c_threshold=100, greyscale_img=True):
    matrix = np.full((img.shape[0],img.shape[1]), 255)
    f = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if greyscale_img:
                m = img[i-1:i+2,j-1:j+2]
                v = abs(np.sum(m*f))
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
                v = abs(np.sum(m*f))
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
    matrix = np.full(img1.shape, 255)

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

def cartesian_to_polar(xy_deriv):
    thetas = []
    radiuses = []
    data = []
    center = (xy_deriv.shape[1]/2, xy_deriv.shape[0]/2)

    for i in range(0, xy_deriv.shape[0]):
        for j in range(0, xy_deriv.shape[1]):
            if xy_deriv[i,j] == 0:
                c = Coordinate(j-center[0],i-center[1])
                thetas.append(c.theta)
                radiuses.append(c.radius)
                data.append(c)

    plt.scatter(thetas, radiuses)
    plt.show()

    return data

def points_of_interest(xy_deriv, iterations=1):
    matrix = xy_deriv
    for i in range(0,iterations):
        matrix = iteration(matrix)

    return matrix

def iteration(img, stride=20):
    dim = (int(img.shape[0]/3), int(img.shape[1]/3))
    min_dim = min(dim[0], dim[1])
    matrix = np.full(img.shape, 255)
    points = []

    for i in range(0, int(img.shape[0]/stride)):
        row = i * stride
        for j in range(0, int(img.shape[1]/stride)):
            col = j * stride
            tot_m, tot_n, c = 0, 0, 0

            for m in range(row, min(img.shape[0], row+dim[0])):
                for n in range(col, min(img.shape[1], col+dim[1])):
                    if img[m,n] == 0:
                        tot_m += m
                        tot_n += n
                        c += 1

            if c > 0:
                m, n = int(tot_m/c),int(tot_n/c)
                matrix[m,n] = 0
                points.append((m,n))

    return matrix, points

def x_deriv_filter(x_deriv):
    matrix = np.full(x_deriv.shape, 0)

    for i in range(0, x_deriv.shape[0]):
        for j in range(0, x_deriv.shape[1]):
            if x_deriv[i,j] == 0:
                break
            matrix[i,j] = 255

    for i in range(0, x_deriv.shape[0]):
        for j in range(x_deriv.shape[1]-1, -1, -1):
            if x_deriv[i,j] == 0:
                break
            matrix[i,j] = 255

    return matrix

def y_deriv_filter(y_deriv):
    matrix = np.full(y_deriv.shape, 0)

    for j in range(0, y_deriv.shape[1]):
        for i in range(0, y_deriv.shape[0]):
            if y_deriv[i,j] == 0:
                break
            matrix[i,j] = 255

    for j in range(0, y_deriv.shape[1]):
        for i in range(y_deriv.shape[0]-1, -1, -1):
            if y_deriv[i,j] == 0:
                break
            matrix[i,j] = 255

    return matrix

def xy_deriv_filter(x_filter, y_filter):
    matrix = np.zeros(x_filter.shape)

    for i in range(0, x_filter.shape[0]):
        for j in range(0, y_filter.shape[1]):
            matrix[i,j] = min(x_filter[i,j], y_filter[i,j])

    return matrix

def mark_edges(xy_deriv):
    matrix = np.zeros(xy_deriv.shape)
    marker = 1
    h = {}   ## marker -> pixel count
    h[0] = -1

    for i in range(0, xy_deriv.shape[0]):
        for j in range(0, xy_deriv.shape[1]):
            v = breadth_first_search(xy_deriv, matrix, i, j, marker)
            if v > 0:
                h[marker] = v
                marker += 1

    return matrix, h

def breadth_first_search(xy_deriv, matrix, r, c, marker):
    points = deque([(r,c)])
    count = 0

    while points:
        p = points.popleft()
        r, c = p[0], p[1]
        if 0 <= r < xy_deriv.shape[0] and 0 <= c < xy_deriv.shape[1]:
            if matrix[r,c] == 0 and xy_deriv[r,c] == 0:
                matrix[r,c] = marker
                count += 1
                points.append((r-1,c)), points.append((r+1,c)), points.append((r,c-1)), points.append((r,c+1))

    return count

def prune_marked_edges(marked_edges, marker_dict, threshold=100):
    matrix = np.full(marked_edges.shape, 255)

    for i in range(0, marked_edges.shape[0]):
        for j in range(0, marked_edges.shape[1]):
            if marker_dict[marked_edges[i,j]] >= threshold:
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

    #x_deriv = x_derivative(gauss_img)
    #y_deriv = y_derivative(gauss_img)
    #xy_deriv = combine_deriv(x_deriv, y_deriv)

    #border_segments = xy_deriv_segments(xy_deriv, segments)
    #polar_data = cartesian_to_polar(xy_deriv)

    x_deriv_color = x_derivative(img, c_threshold=85, greyscale_img=False)
    y_deriv_color = y_derivative(img, c_threshold=85, greyscale_img=False)
    xy_deriv_color = combine_deriv(x_deriv_color, y_deriv_color)
    marked_edges, marker_count_dict = mark_edges(xy_deriv_color)
    pruned_edges = prune_marked_edges(marked_edges, marker_count_dict)

    x_filter = x_deriv_filter(x_deriv_color)
    y_filter = y_deriv_filter(y_deriv_color)
    xy_filter = xy_deriv_filter(x_filter, y_filter)
    #poi_color_img, _ = points_of_interest(xy_deriv_color)


    #cv2.imwrite("./outputs/birdo_gauss.png", gauss_img)
    #cv2.imwrite("./outputs/birdo_x_deriv.png", x_deriv)
    #cv2.imwrite("./outputs/birdo_y_deriv.png", y_deriv)
    cv2.imwrite("./outputs/birdo_xy_filter.png", xy_filter)
    cv2.imwrite("./outputs/birdo_xy_deriv_color.png", xy_deriv_color)
    cv2.imwrite("./outputs/birdo_marked_edges.png", pruned_edges)
    #cv2.imwrite("./outputs/birdo_xy_deriv.png", xy_deriv)
    #cv2.imwrite("./outputs/points_of_interest_color.png", poi_color_img)



if __name__ == '__main__':
    derivative_analysis()
