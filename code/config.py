import numpy as np

"""
This class contain all the hyperparameter that we have to tune to get the best result 
"""
class Config:
    def __init__(self):
        #CAMERA CALIBRATION 
        self.CHESSBOARD_SIZE = (9, 6)

        #COLOR AND GRADIENT SPACE
        self.SOBEL_KERNEL_SIZE = 15
        self.ABSOLUTE_GRADIENT_THRESHOLD = (30, 100)
        self.MAGNITUDE_GRADIENT_THRESHOLD = (30, 100)
        self.DIRECTION_GRADIENT_THRESHOLD = (0.7, 1.3)
        self.COLOR_CHANNEL = 'S'
        self.COLOR_CHANNEL_THRESHOLD = (170, 255)

        #PERSPECTIVE TRANSFORMATION
        self.PERSPECTIVE_SRC = np.float32([
            [580, 460],
            [205, 720],
            [1110, 720],
            [703, 460]
        ])
        self.PERSPECTIVE_DST = np.float32([
            [250, 0],
            [250, 720],
            [960, 720],
            [960, 0]
        ])

        #FINAL LANE DETECTION USING SLIDING WINDOW
        self.NUMBER_OF_WINDOWS = 9
        self.WINDOW_MARGIN = 100
        self.MINIMUM_PIXEL_IN_WINDOW = 50
        self.METER_PER_PIXEL_X = 3.7/700
        self.METER_PER_PIXEL_Y = 32.0/720
        self.N_FRAMES = 25