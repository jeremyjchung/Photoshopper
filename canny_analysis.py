import cv2
import fractions
import math
import numpy as np

from collections import deque
from scipy.spatial import Delaunay
from skimage.segmentation import slic

def canny_edge_detector(grey_img, threshold1=75, threshold2=150):
    return cv2.Canny(grey_img, threshold1, threshold2)

def detect_end_points(img):
    matrix = np.zeros(img.shape)
    points = []

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] > 0:
                l, r = int(img[i, max(j-1,0)]), int(img[i,min(j+1, img.shape[1]-1)])
                u, d = int(img[max(i-1, 0),j]), int(img[min(i+1, img.shape[0]-1),j])

                if l + r + u + d <= 255:
                    matrix[i,j] = 255
                    points.append((i,j))

    return matrix, points

def dilation(img):
    img_plus_border = np.zeros((img.shape[0]+2, img.shape[1]+2))
    img_plus_border[1:img_plus_border.shape[0]-1, 1:img_plus_border.shape[1]-1] = np.array(img)

    for i in range(2, img_plus_border.shape[0]-2):
        for j in range(2, img_plus_border.shape[1]-2):
            if img_plus_border[i,j] > 0:
                for m in range(i-2, i+3):
                    for n in range(j-2, j+3):
                        img_plus_border[m,n] = 255

    return img_plus_border[1:img_plus_border.shape[0]-1, 1:img_plus_border.shape[1]-1]

def guess_edge_points(end_points, edge_img, img, window=10):
    matrix = np.array(edge_img)
    dilation_matrix = dilation(edge_img)

    for i in range(0, len(end_points)):
        r, c = end_points[i][0], end_points[i][1]
        p = [r,c,img[r,c]]
        best_point, best_v = None, 100000000000
        c_window = min(r, c, img.shape[0]-r-1, img.shape[1]-c-1, window)

        right_c, left_c = c+c_window, c-c_window
        up_r, down_r = r-c_window, r+c_window

        for i in range(r-c_window, r+c_window):
            p1, p2 = np.array([i,right_c,img[i,right_c]]), np.array([i,left_c,img[i,left_c]])
            v1, v2 = np.linalg.norm(p-p1), np.linalg.norm(p-p2)

            if dilation_matrix[p1[0],p1[1]] == 0 and v1 < best_v:
                best_v, best_point = v1, (i,right_c)
            if dilation_matrix[p2[0],p2[1]] == 0 and v2 < best_v:
                best_v, best_point = v2, (i,left_c)

        for j in range(c-c_window, c+c_window):
            p1, p2 = np.array([up_r,j,img[up_r,j]]), np.array([down_r,j,img[down_r,j]])
            v1, v2 = np.linalg.norm(p-p1), np.linalg.norm(p-p2)

            if dilation_matrix[p1[0],p1[1]] == 0 and v1 < best_v:
                best_v, best_point = v1, (up_r,j)
            if dilation_matrix[p2[0],p2[1]] == 0 and v2 < best_v:
                best_v, best_point = v2, (down_r,j)

        if best_point != None:
            end_points.append(best_point)
            matrix[best_point[0],best_point[1]] = 255

    return matrix

# http://www.cvc.uab.es/~asappa/publications/C__SITIS_2006.pdf
# end_points = nx2 array, img = grey scale img
def global_linking(end_points, img):
    tri = Delaunay(np.array(end_points))
    simplices = tri.simplices
    graph = np.full((len(end_points), len(end_points)), -1)

    for i in range(0, simplices.shape[0]):
        v1, v2, v3 = simplices[i,0], simplices[i,1], simplices[i,2]
        p1, p2, p3 = end_points[v1], end_points[v2], end_points[v3]
        f1, f2, f3 = np.array([p1[0],p1[1],img[p1[0],p1[1]]]), np.array([p2[0],p2[1],img[p2[0],p2[1]]]), np.array([p3[0],p3[1],img[p3[0],p3[1]]])

        graph[v1,v2], graph[v1,v3] = np.linalg.norm(f1-f2), np.linalg.norm(f1-f3)
        graph[v2,v1], graph[v2,v3] = graph[v1,v2], np.linalg.norm(f2-f3)
        graph[v3,v1], graph[v3,v2] = graph[v1,v3], graph[v2,v3]

    mst = np.full(graph.shape, -1)
    n = int(len(end_points) * (len(end_points)-1) / 2)
    edges = np.zeros((n), dtype=[('node1', int), ('node2', int), ('distance', float)])

    for i in range(0, graph.shape[0]):
        for j in range(0, i):
            n -= 1
            edges[n] = (i, j, graph[i,j])

    edges = np.sort(edges, order='distance')
    seen = {}

    for i in range(0, edges.shape[0]):
        edge = edges[i]
        n1, n2, d = edge[0], edge[1], edge[2]

        if d < 0:
            continue
        if n1 in seen and n2 in seen:
            continue

        mst[n1,n2], mst[n2,n1] = d, d
        seen[n1], seen[n2] = None, None

    global_img = np.zeros(img.shape)

    for i in range(0, mst.shape[0]):
        for j in range(0, i):
            if mst[i,j] >= 0:
                p1, p2 = end_points[i], end_points[j]
                global_img[p1[0],p1[1]], global_img[p2[0],p2[1]] = 255, 255

                if p2[1] == p1[1]:
                    y1, y2 = int(min(p1[0],p2[0])), int(max(p1[0],p2[0])) + 1
                    for y in range(y1, y2):
                        global_img[y,int(p1[1])] = 255
                else:
                    slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
                    intercept = p1[0] - slope * p1[1]
                    f = lambda x: slope * x + intercept
                    x1, x2 = int(min(p1[1],p2[1])) + 1, int(max(p1[1],p2[1])) + 1

                    ## color grid spaces that the line passes through
                    ## look at every y-value for every n*0.2 x-value
                    for x in range(x1*10, x2*10):
                        v = f(x/10)
                        if v >= 0 and v < global_img.shape[0]:
                            global_img[int(v),int(x/10)] = 255

    return global_img

def combine_imgs(img1, img2):
    matrix = np.zeros(img1.shape)

    for i in range(0, img1.shape[0]):
        for j in range(0, img1.shape[1]):
            matrix[i,j] = max(img1[i,j], img2[i,j])

    return matrix

# connect pixels that are diagonally adjacent so that they are horizontally/vertically reachable
def connect_diagonal_pixels(img):
    img_plus_border = np.zeros((img.shape[0]+2, img.shape[1]+2))
    img_plus_border[1:img_plus_border.shape[0]-1,1:img_plus_border.shape[1]-1] = img

    for i in range(1, img_plus_border.shape[0]-1):
        for j in range(1, img_plus_border.shape[1]-1):
            if img_plus_border[i,j] > 0:
                u, r, d, l = img_plus_border[i-1,j], img_plus_border[i,j+1], img_plus_border[i+1,j], img_plus_border[i,j-1]
                ul, ur = img_plus_border[i-1,j-1], img_plus_border[i-1,j+1]
                dl, dr = img_plus_border[i+1,j-1], img_plus_border[i+1,j+1]

                if ul > 0 and u == 0 and l == 0:
                    img_plus_border[i-1,j] = 255
                if ur > 0 and u == 0 and r == 0:
                    img_plus_border[i-1,j] = 255
                if dl > 0 and d == 0 and l == 0:
                    img_plus_border[i+1,j] = 255
                if dr > 0 and d == 0 and r == 0:
                    img_plus_border[i+1,j] = 255

    return img_plus_border[1:img_plus_border.shape[0]-1,1:img_plus_border.shape[1]-1]

def global_iterations(canny_img, img, iterations=10, start_guess=6):
    final_img = np.array(canny_img)
    end_points_img, end_points = None, None

    for i in range(0, iterations):
        final_img = connect_diagonal_pixels(final_img)
        end_points_img, end_points = detect_end_points(final_img)

        if i >= start_guess:
            final_img = guess_edge_points(end_points, final_img, img)

        global_linking_img = global_linking(end_points, img)
        final_img = combine_imgs(final_img, global_linking_img)

    final_img = fill_details(final_img)
    return final_img

def fill_details(edge_img):
    matrix = np.array(edge_img)

    for i in range(1, edge_img.shape[0]-1):
        for j in range(1, edge_img.shape[1]-1):
            if edge_img[i,j] == 0:
                if edge_img[i-1,j] > 0 and edge_img[i+1,j] > 0:
                    matrix[i,j] = 255
                elif edge_img[i,j-1] > 0 and edge_img[i,j+1] > 0:
                    matrix[i,j] = 255

    return matrix

def blacken_background(edge_img):
    matrix = np.full(edge_img.shape, 255)
    seen = np.zeros(edge_img.shape)
    queue = deque([(0,0), (edge_img.shape[0]-1,0), (0,edge_img.shape[1]-1), (edge_img.shape[0]-1,edge_img.shape[1]-1)])

    while queue:
        p = queue.popleft()

        if p[0] < 0 or p[1] < 0 or p[0] >= edge_img.shape[0] or p[1] >= edge_img.shape[1]:
            continue
        if seen[p[0],p[1]] == 1:
            continue
        if edge_img[p[0],p[1]] == 0:
            matrix[p[0],p[1]] = 0
            queue.append((p[0],p[1]+1)), queue.append((p[0],p[1]-1))
            queue.append((p[0]-1,p[1])), queue.append((p[0]+1,p[1]))

        seen[p[0],p[1]] = 1

    return matrix

def prune_edges(edge_img):
    labels = np.zeros(edge_img.shape)
    matrix = np.zeros(edge_img.shape)
    h = {}
    count = 1
    h[count] = 0

    for i in range(0, edge_img.shape[0]):
        for j in range(0, edge_img.shape[1]):
            queue = deque([(i,j)])

            while queue:
                p = queue.popleft()

                if p[0] < 0 or p[1] < 0 or p[0] >= edge_img.shape[0] or p[1] >= edge_img.shape[1]:
                    continue
                if labels[p[0],p[1]] > 0 or edge_img[p[0],p[1]] == 0:
                    continue

                h[count] += 1
                labels[p[0],p[1]] = count
                queue.append((p[0],p[1]+1)), queue.append((p[0],p[1]-1))
                queue.append((p[0]-1,p[1])), queue.append((p[0]+1,p[1]))

            if h[count] > 0:
                count += 1
                h[count] = 0

    best_label, best_count = 1, h[1]

    for k, v in h.items():
        if v > best_count:
            best_count = v
            best_label = k

    for i in range(0, edge_img.shape[0]):
        for j in range(0, edge_img.shape[1]):
            if labels[i,j] == best_label:
                matrix[i,j] = 255

    return matrix

## not effective ....
def segmentation_prune(filled_img, img, threshold=0.15, n=120, c=40):
    matrix = np.zeros(filled_img.shape)
    segments = slic(img, n_segments=n, compactness=c)
    h = {}    ## segment label => (total pixels, count of 255 pixels)

    for i in range(0, segments.shape[0]):
        for j in range(0, segments.shape[1]):
            s = segments[i,j]
            if s not in h:
                h[s] = np.array([0,0])
            if filled_img[i,j] > 0:
                h[s][1] += 1
            h[s][0] += 1

    for i in range(0, filled_img.shape[0]):
        for j in range(0, filled_img.shape[1]):
            if filled_img[i,j] > 0:
                s = segments[i,j]
                if h[s][1] / h[s][0] >= threshold:
                    matrix[i,j] = 255

    return matrix

def analysis_v1():
    img = cv2.imread("./images/doggo.jpg", 0)
    canny_img = canny_edge_detector(img)
    closed_shape = global_iterations(canny_img, img)
    pruned_edges = prune_edges(closed_shape)
    filled_in = blacken_background(pruned_edges)

    cv2.imwrite("./outputs/v1/canny.png", canny_img)
    cv2.imwrite("./outputs/v1/closed.png", closed_shape)
    cv2.imwrite("./outputs/v1/closed_pruned.png", pruned_edges)
    cv2.imwrite("./outputs/v1/filled.png", filled_in)

if __name__ == '__main__':
    analysis_v1()
