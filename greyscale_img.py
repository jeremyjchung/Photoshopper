import numpy as np

def generate(img):
    matrix = np.zeros((img.shape[0], img.shape[1]))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            ### https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
            ### simplified version
            brightness = (2 * img[i,j,0] + img[i,j,1] + 3 * img[i,j,2]) / 6
            matrix[i,j] = brightness

    return matrix
