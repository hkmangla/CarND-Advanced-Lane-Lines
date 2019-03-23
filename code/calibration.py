import numpy as np
import cv2
import glob

class CameraCalibration:

    def __init__(self, calib_image_dir, chessboard_size):
        self.chessboard_size = chessboard_size
        self.calib_image_dir = calib_image_dir
        self.objectPoints = []
        self.imagePoints = []

        self.setCalibrationPoints()

    def setCalibrationPoints (self):
        
        # define 3d object point of chessboard_size 
        objectP = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), np.float32)
        objectP[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        images = glob.glob(self.calib_image_dir + '*')

        for fname in images:
            image = cv2.imread(fname)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            #find corners from the chessboard images
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret == True:
                self.objectPoints.append(objectP)
                self.imagePoints.append(corners)

                cv2.drawChessboardCorners(image, self.chessboard_size, corners, ret)

    def undistortImage (self, image):
        
        imageSize = (image.shape[1], image.shape[0])
        #compute camera matrix and distortion coefficients
        ret, camera_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.objectPoints, self.imagePoints, imageSize, None, None)

        undistorted_image = cv2.undistort(image, camera_matrix, distCoeffs, None, camera_matrix)

        return undistorted_image
