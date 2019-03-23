import numpy as np
import cv2

class BirdEye:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def warper (self, image):

        M = cv2.getPerspectiveTransform(self.src, self.dst)
        M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        image_size = (image.shape[1], image.shape[0])

        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR), M_inv
