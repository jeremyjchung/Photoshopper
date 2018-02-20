import numpy as np

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta = np.arctan2(y,x)
        self.radius = (x**2 + y**2)**0.5
