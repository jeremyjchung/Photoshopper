import cv2
import numpy as np

from sklearn.cluster import KMeans

# data -> feature vector: [r, g, b, row, col]
def get_data(img):
    data = np.zeros((img.shape[0]*img.shape[1],img.shape[2] + 2))
    c = 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            data[c,0:3] = img[i,j,:]
            data[c,3] = i
            data[c,4] = j
            c += 1

    return data

def kmeans_clustering(data, img):
    bow = KMeans(n_clusters=100, random_state=0).fit(data)
    predictions = bow.predict(data)
    centers = bow.cluster_centers_

    matrix = np.zeros(img.shape)
    c = 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            matrix[i,j,:] = centers[predictions[c],0:3]
            c += 1

    return matrix

def clustering_analysis():
    img = cv2.imread("./images/birdo.jpg", cv2.IMREAD_UNCHANGED)
    bow_img = kmeans_clustering(get_data(img), img)

    cv2.imwrite("./outputs/birdo_clustering.png", bow_img)

f __name__ == '__main__':
    clustering_analysis()
