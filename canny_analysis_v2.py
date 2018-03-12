import cv2
import numpy as np
import math

from collections import deque
from scipy.spatial import Delaunay

def compute_gradient_angle(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    matrix = np.zeros(img.shape)

    for i in range(0, sobelx.shape[0]):
        for j in range(0, sobelx.shape[1]):
            matrix[i,j] = math.degrees(math.atan2(sobely[i,j],sobelx[i,j]))

    return matrix

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

def canny_edge_detector(grey_img, threshold1=75, threshold2=150):
    return cv2.Canny(grey_img, threshold1, threshold2)

def canny_edge_connector(edge_img):
    matrix = np.array(edge_img)

    for i in range(1, edge_img.shape[0]-1):
        for j in range(1, edge_img.shape[1]-1):
            if edge_img[i,j] == 0:
                left, right, up, down = False, False, False, False

                for r in range(i-1, i+2):
                    if edge_img[r,j-1] > 0: left = True
                    if edge_img[r,j+1] > 0: right = True
                if left and right:
                    matrix[i,j] = 255
                    continue

                for c in range(j-1, j+2):
                    if edge_img[i-1,c] > 0: up = True
                    if edge_img[i+1,c] > 0: down = True
                if up and down:
                    matrix[i,j] = 255
                    continue

    return matrix

def find_endpoints(edge_img, win=8):
    matrix = np.zeros(edge_img.shape)
    seen = np.zeros(edge_img.shape)
    h = {}    ## label => connected endpoints

    for i in range(win, edge_img.shape[0]-win):
        for j in range(win, edge_img.shape[1]-win):
            if edge_img[i,j] > 0:
                row, col = np.zeros(2*win+1), np.zeros(2*win+1)
                for m in range(i-win, i+win+1):
                    for n in range(j-win, j+win+1):
                        if edge_img[m,n] > 0:
                            row[m-i], col[n-j] = 1, 1

                if np.sum(row) < 2*win and np.sum(col) < 2*win:
                    matrix[i,j] = 255

    count = 0
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            count += 1
            points = deque([(i,j)])
            while points:
                p = points.popleft()
                r, c = p[0], p[1]

                if r < 0 or c < 0 or r >= matrix.shape[0] or c >= matrix.shape[1]:
                    continue
                if matrix[r,c] == 0 or seen[r,c] == 1:
                    continue
                if count not in h:
                    h[count] = []

                h[count].append((r,c))
                points.append((r+1,c)), points.append((r-1,c)), points.append((r,c+1)), points.append((r,c-1))
                points.append((r+1,c+1)), points.append((r+1,c-1)), points.append((r-1,c+1)), points.append((r-1,c-1))
                seen[r,c] = 1

    matrix = np.zeros(edge_img.shape)
    endpoints = []
    for k, v in h.items():
        r, c = 0, 0
        for i in range(0, len(v)):
            r += v[i][0]
            c += v[i][1]

        r, c = int(r/len(v)), int(c/len(v))
        matrix[r,c] = 255
        endpoints.append((r,c))

    return matrix, endpoints

def mst(img, gradient, endpoints, edge_img):
    tri = Delaunay(np.array(endpoints))
    simplices = tri.simplices
    matrix = np.array(edge_img)
    h = {}
    links = []    ## graph edges

    for i in range(0, simplices.shape[0]):
        v1, v2, v3 = simplices[i,0], simplices[i,1], simplices[i,2]
        p1, p2, p3 = endpoints[v1], endpoints[v2], endpoints[v3]

        f1 = np.array([p1[0], p1[1], img[p1[0],p1[1]], gradient[p1[0],p1[1]]])
        f2 = np.array([p2[0], p2[1], img[p2[0],p2[1]], gradient[p2[0],p2[1]]])
        f3 = np.array([p3[0], p3[1], img[p3[0],p3[1]], gradient[p3[0],p3[1]]])

        #f1 = np.array([p1[0], p1[1], img[p1[0],p1[1]]])
        #f2 = np.array([p2[0], p2[1], img[p2[0],p2[1]]])
        #f3 = np.array([p3[0], p3[1], img[p3[0],p3[1]]])

        links.append((v1,v2,np.linalg.norm(f1-f2)))
        links.append((v2,v3,np.linalg.norm(f3-f2)))
        links.append((v1,v3,np.linalg.norm(f3-f1)))

    #for i in range(0, len(endpoints)):
    #    for j in range(i+1, len(endpoints)):
    #        r1, c1, r2, c2 = endpoints[i][0], endpoints[i][1], endpoints[j][0], endpoints[j][1]
    #        f1, f2 = np.array([r1, c1, img[r1,c1], img[r1,c1]]), np.array([r2, c2, img[r2,c2], img[r2,c2]])
    #        links.append((i,j,np.linalg.norm(f1-f2)))

    links.sort(key=lambda tup: tup[2])
    sub_graphs = []
    seen = set()

    for l in range(0, int(0.5*len(links))):
        link = links[l]
        portions = []
        for i in range(0, len(sub_graphs)):
            g = sub_graphs[i]
            if link[0] in g and link[1] in g:
                break
            if link[0] in g or link[1] in g:
                portions.append(i)

        if len(portions) == 0:
            if link[0] not in seen:
                sub_graphs.append(set([link[0],link[1]]))
                connect_two_points(endpoints[link[0]], endpoints[link[1]], matrix)
            else:
                connect_two_points(endpoints[link[0]], endpoints[link[1]], matrix)
        elif len(portions) == 1:
            sub_graphs[portions[0]].add(link[0]), sub_graphs[portions[0]].add(link[1])
            connect_two_points(endpoints[link[0]], endpoints[link[1]], matrix)
        elif len(portions) == 2:
            g = sub_graphs[portions[0]].union(sub_graphs[portions[1]])
            sub_graphs.pop(max(portions)), sub_graphs.pop(min(portions))
            sub_graphs.append(g)
            connect_two_points(endpoints[link[0]], endpoints[link[1]], matrix)

        seen.add(link[0]), seen.add(link[1])

    return matrix

def connect_two_points(p1, p2, matrix):
    if p2[1] == p1[1]:
        y1, y2 = int(min(p1[0],p2[0])), int(max(p1[0],p2[0])) + 1
        for y in range(y1, y2):
            matrix[y,int(p1[1])] = 255
    else:
        slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
        intercept = p1[0] - slope * p1[1]
        f = lambda x: slope * x + intercept
        x1, x2 = int(min(p1[1],p2[1])) + 1, int(max(p1[1],p2[1])) + 1

        ## color grid spaces that the line passes through
        ## look at every y-value for every n*0.2 x-value
        for x in range(x1*10, x2*10+10):
            v = f(x/10)
            if v >= 0 and v < matrix.shape[0] and x/10 < matrix.shape[1]:
                matrix[int(v),int(x/10)] = 255

def analysis_v2():
    img = cv2.imread("./images/owl.jpg", 0)
    gradient = compute_gradient_angle(img)
    canny_img = canny_edge_detector(img)
    canny_connected = canny_edge_connector(canny_img)
    #endpoint_img, endpoints = find_endpoints(canny_connected)
    endpoint_img, endpoints = detect_end_points(canny_connected)
    mst_img = mst(img, gradient, endpoints, canny_connected)

    cv2.imwrite("./outputs/v2/canny.png", canny_img)
    #cv2.imwrite("./outputs/v2/connected.png", canny_connected)
    cv2.imwrite("./outputs/v2/endpoints.png", endpoint_img)
    cv2.imwrite("./outputs/v2/mst.png", mst_img)

if __name__ == '__main__':
    analysis_v2()
