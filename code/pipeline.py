import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from config import Config
from calibration import CameraCalibration
from bird_eye import BirdEye
from color_and_gradient_thresholding import ColorGradientSpace
from lane_detection import LaneDetect

"""
Build lane line detection pipeline using the above classes
"""
class AdvancedLaneDetection ():
    def __init__(self):
        self.config = Config()
        self.frame_count = 0

        self.left_rc, self.right_rc = (None, None)
        self.vehicle_position_text = ''
        self.cameraCalibration = CameraCalibration('../camera_cal/', self.config.CHESSBOARD_SIZE)

        self.colorGradientSpace = ColorGradientSpace(self.config.ABSOLUTE_GRADIENT_THRESHOLD, self.config.MAGNITUDE_GRADIENT_THRESHOLD,
                    self.config.DIRECTION_GRADIENT_THRESHOLD, self.config.COLOR_CHANNEL_THRESHOLD, self.config.SOBEL_KERNEL_SIZE, self.config.COLOR_CHANNEL)
                    
        self.perspectiveTransform = BirdEye(self.config.PERSPECTIVE_SRC, self.config.PERSPECTIVE_DST)
        
        self.laneDetect = LaneDetect(self.config.NUMBER_OF_WINDOWS, self.config.WINDOW_MARGIN, self.config.MINIMUM_PIXEL_IN_WINDOW,
                                 self.config.METER_PER_PIXEL_X, self.config.METER_PER_PIXEL_Y, self.config.N_FRAMES)

    def pipeline (self, image):

        undist_image = self.cameraCalibration.undistortImage(image)

        warped_image, M_inv = self.perspectiveTransform.warper(
                        self.colorGradientSpace.colorGradientCombined(undist_image))

        left_fitx, right_fitx, ploty = self.laneDetect.detectLine(warped_image)

        # print('Lane detected!')
        output_image = self.map_lane(image, undist_image, warped_image,
                            left_fitx, right_fitx, ploty, M_inv)

        # print(warped_image.shape)
        # warped_image = cv2.resize(warped_image,(640,360))
        # warped_image = np.hstack((warped_image, warped_image))
        # print(warped_image.shape)
        warped_image_3d = np.zeros_like(output_image)
        warped_image_3d[:, :, 0] = warped_image
        output_image = np.vstack((warped_image_3d, output_image))
        if (self.frame_count%10 == 0):
            self.left_rc, self.right_rc = self.laneDetect.get_curvature(ploty)
            self.vehicle_position_text = self.laneDetect.get_vehicle_position_text()

        output_image = self.put_text(output_image, 'Curvature: {0:.2f}m'.format(round((self.left_rc+self.right_rc)/2,2)), (25, 800))            
        output_image = self.put_text(output_image, 'Vehicle Position: ' + self.vehicle_position_text, (25, 850))            
        self.frame_count += 1
        return output_image

    def map_lane (self, image, undist, warped, left_fitx, right_fitx, ploty, Minv):

        warp_zero = np.uint8(np.zeros_like(warped))
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        return result

    def put_text (self, image, text, position):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,0,0)
        lineType               = 2

        cv2.putText(image, text, 
            position, 
            font, 
            fontScale,
            fontColor,
            lineType)

        return image

advancedLaneDetection = AdvancedLaneDetection()

project_video_output_fname = '../output_video/project_video_output1.mp4'
project_video = VideoFileClip("../project_video.mp4").subclip(10, 30)

project_video_output = project_video.fl_image(advancedLaneDetection.pipeline)
project_video_output.write_videofile(project_video_output_fname, audio=False)

# image = cv2.imread('../calibrated_test_images/test1.jpg')

# output_image = advancedLaneDetection.pipeline(image)

# import matplotlib.pyplot as plt
# print(output_image.shape)
# plt.imshow(output_image)
# plt.show()