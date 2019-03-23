import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#HyperParameters

#CAMERA CALIBRATION 
CHESSBOARD_SIZE = (9, 6)

#COLOR AND GRADIENT SPACE
SOBEL_KERNEL_SIZE = 15
ABSOLUTE_GRADIENT_THRESHOLD = (30, 100)
MAGNITUDE_GRADIENT_THRESHOLD = (30, 100)
DIRECTION_GRADIENT_THRESHOLD = (np.pi/4, np.pi/2)
COLOR_CHANNEL = 2
COLOR_CODE = 'HLS'
COLOR_CHANNEL_THRESHOLD = (170, 255)

#PERSPECTIVE TRANSFORMATION
PERSPECTIVE_SRC = np.float32([
    # [520, 500],
    # [225, 700],
    # [1075, 700],
    # [760, 500]
    [580, 460],
    [205, 720],
    [1110, 720],
    [703, 460]
])
PERSPECTIVE_DST = np.float32([
    [200, 0],
    [200, 720],
    [1050, 720],
    [1050, 0]
])

#FINAL LANE DETECTION USING SLIDING WINDOW
NUMBER_OF_WINDOWS = 9
WINDOW_MARGIN = 100
MINIMUM_PIXEL_IN_WINDOW = 50
METER_PER_PIXEL_X = 3.7/700
METER_PER_PIXEL_Y = 32.0/720
N_FRAMES = 20

class Utils:
    def __init__(self):
        pass

    def showImages(self, image1, image2):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(image1)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(image2)
        ax2.set_title('Undistorted Image', fontsize=30)

        plt.show()
    
    def applyFilter (self, input_directory, output_directory, filter):

        images = glob.glob(input_directory + '*')
        for fname in images:
            image = cv2.imread(fname)
            filtered_image = filter(image)

            # utils.showImages(image, filtered_image)

            print('Saving filtered image ' + fname.split('/')[1] + ' ' + str(filtered_image.shape) + ' into ' + output_directory + ' folder')
            cv2.imwrite(output_directory + fname.split('/')[1], filtered_image)

    def drawOnResults (self, image, undist, warped, left_fitx, right_fitx, ploty, Minv):

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

#Step 1 Camera Calibration
class CameraCalibration:

    def __init__(self, calib_image_dir, chessboard_size):
        self.chessboard_size = chessboard_size
        self.calib_image_dir = calib_image_dir
        self.objectPoints = []
        self.imagePoints = []

        self.setCalibrationPoints()

    def setCalibrationPoints (self):
        
        objectP = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), np.float32)
        objectP[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        images = glob.glob(self.calib_image_dir + '*')

        for fname in images:
            image = cv2.imread(fname)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret == True:
                self.objectPoints.append(objectP)
                self.imagePoints.append(corners)

                cv2.drawChessboardCorners(image, self.chessboard_size, corners, ret)

    def undistortImage (self, image):
        
        imageSize = (image.shape[1], image.shape[0])
        ret, camera_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.objectPoints, self.imagePoints, imageSize, None, None)

        undistorted_image = cv2.undistort(image, camera_matrix, distCoeffs, None, camera_matrix)

        return undistorted_image


# Step2: Color and gradient spaces
class ColorGradientSpace:
    def __init__ (self, absGradThreshold, magGradThreshold, dirGradThreshold, colorThreshold=(0,255), kernel_size=3, colorChannel='S', colorCode='HLS'):
        self.absGradThreshold = absGradThreshold
        self.magGradThreshold = magGradThreshold
        self.dirGradThreshold = dirGradThreshold
        self.colorThreshold = colorThreshold
        colorDict = {'HLS': cv2.COLOR_BGR2HLS, 'HSV': cv2.COLOR_BGR2HSV, 'LAB': cv2.COLOR_BGR2LAB}
        self.colorChannel = colorChannel
        self.colorCode = colorDict[colorCode]
        self.sobel_kernel = kernel_size

    def absoluteGradient (self, image, orient='x', thresh=(0,255)):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel = None
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        abs_sobel = np.absolute(sobel)
        scaled_img = np.uint8(255.0*abs_sobel/np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_img)
        grad_binary[(scaled_img <= thresh[1]) & (scaled_img >= thresh[0])] = 255
        
        return grad_binary

    def magnitudeGradient (self, image, thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        abs_sobel = np.sqrt(sobelx**2, sobely**2)
        scaled_img = np.uint8(255*abs_sobel/np.max(abs_sobel))

        mag_binary = np.zeros_like(scaled_img)
        mag_binary[(scaled_img <= thresh[1]) & (scaled_img >= thresh[0])] = 255
        return mag_binary

    def directionGradient (self, image, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        scaled_img = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        dir_binary = np.zeros_like(scaled_img)
        dir_binary[(scaled_img <= thresh[1]) & (scaled_img >= thresh[0])] = 255

        return dir_binary

    def gradientCombined (self, image):

        gradx = self.absoluteGradient(image, orient='x', thresh=self.absGradThreshold)
        grady = self.absoluteGradient(image, orient='y', thresh=self.absGradThreshold)
        mag_binary = self.magnitudeGradient(image, thresh=self.magGradThreshold)
        dir_binary = self.directionGradient(image, thresh=self.dirGradThreshold)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 255) & (grady == 255)) | ((mag_binary == 255) & (dir_binary == 255))] = 255

        return combined

    def compute_hls_white_yellow_binary (self, image):
        hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                 & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                 & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                ] = 1

        img_hls_white_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                 & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                 & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                ] = 1
        
        img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 255

        return img_hls_white_yellow_bin

    def colorChannelConversion (self, image):
        colorImage = cv2.cvtColor (image, self.colorCode)

        channel_image = colorImage[:, :, self.colorChannel]
        binary = np.zeros_like(channel_image)
        binary[(channel_image <= self.colorThreshold[1]) & (channel_image >= self.colorThreshold[0])] = 255

        return binary

    def colorGradientCombined (self, image):
        # colored = self.compute_hls_white_yellow_binary(image)
        colored = self.colorChannelConversion(image)
        # l_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        gradientCombined = self.gradientCombined(image)

        combined = np.zeros_like(gradientCombined)
        combined[(colored == 255) | (gradientCombined == 255)] = 255

        return combined

#step 3 Perspective Transform
class PerspectiveTransform:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def warped (self, image):

        M = cv2.getPerspectiveTransform(self.src, self.dst)
        M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        image_size = (image.shape[1], image.shape[0])

        return cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    
# step 4 to detect lanes

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None 
        #polynomial coefficients for the last n fit
        self.recent_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

class LaneDetect():
    def __init__(self, n_windows, margin, minpix, xm_per_pixel, ym_per_pixel, n_frames):
        self.n_windows = n_windows
        self.margin = margin
        self.minpix = minpix
        self.xm_per_pixel = xm_per_pixel
        self.ym_per_pixel = ym_per_pixel
        self.n_frames = n_frames
        self.leftLine = Line()
        self.rightLine = Line()

    def find_lane_pixels(self, image):
        left_lane_inds = []
        right_lane_inds = []
        window_height = np.int(image.shape[0] // self.n_windows)
        
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        out_img = np.dstack((image, image, image))

        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        for window in range(self.n_windows):
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height

            win_leftx_low = leftx_current - self.margin
            win_leftx_high = leftx_current + self.margin
            win_rightx_low = rightx_current - self.margin
            win_rightx_high = rightx_current + self.margin 

            cv2.rectangle(out_img, (win_leftx_low, win_y_low), (win_leftx_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_rightx_low, win_y_low), (win_rightx_high, win_y_high), (0, 255, 0), 2)

            left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_leftx_low) &  (nonzerox < win_leftx_high)).nonzero()[0]
            right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_rightx_low) &  (nonzerox < win_rightx_high)).nonzero()[0]

            left_lane_inds.append(left_inds)
            right_lane_inds.append(right_inds)

            if len(left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[left_inds]))
            if len(right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[right_inds]))
        
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)    
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]    
        
        self.leftLine.allx = leftx
        self.leftLine.ally = lefty
        self.rightLine.allx = rightx
        self.rightLine.ally = righty

        return leftx, lefty, rightx, righty, out_img
        
    def fit_polynomial (self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels (image)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        cv2.polylines(out_img, [np.array(zip(left_fitx, ploty), np.int32)], True, (0,255,255), 3)
        cv2.polylines(out_img, [np.array(zip(right_fitx, ploty), np.int32)], True, (0,255,255), 3)

        return out_img

    def get_curvature (self, ploty):
        
        y_eval = np.max(ploty)
        left_fit_real = np.polyfit(self.leftLine.ally*self.ym_per_pixel, self.leftLine.allx*self.xm_per_pixel, 2)
        right_fit_real = np.polyfit(self.rightLine.ally*self.ym_per_pixel, self.rightLine.ally*self.xm_per_pixel, 2)

        left_curverad = ((1 + (2*left_fit_real[0]*y_eval*self.ym_per_pixel + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
        right_curverad = ((1 + (2*right_fit_real[0]*y_eval*self.ym_per_pixel + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])

        return left_curverad, right_curverad

    def updateLine (self, line, fitx, fit, radius_of_curvature):
        
        last_fit = None
        if line.detected:
            last_fit = line.current_fit
        
        line.current_fit = fit
        if last_fit:
            line.diffs = last_fit - line.current_fit
        
        line.recent_fits.append(line.current_fit)
        line.recent_xfitted.append(fitx)
        if len(line.recent_xfitted) > self.n_frames:
            line.recent_xfitted = line.recent_xfitted[1:]
            line.recent_fits = line.recent_fits[1:]

        line.bestx = np.mean(line.recent_xfitted, axis=0)
        line.best_fit = np.mean(line.recent_fits, axis=0)
        line.radius_of_curvature = radius_of_curvature
        
    def detectLine (self, image):

        left_fit, right_fit, left_fitx, right_fitx, ploty = self.fit_polynomial (image)
        left_rc, right_rc = self.get_curvature(ploty)
        
        # if abs(left_rc - right_rc) > 200:
        #     print('Lanes are not parallel!')

        self.leftLine.line_base_pos = (image.shape[1]//2 - np.mean(self.leftLine.allx))*self.xm_per_pixel
        self.rightLine.line_base_pos = (np.mean(self.rightLine.allx - image.shape[1]//2))*self.xm_per_pixel

        self.updateLine(self.leftLine, left_fitx, left_fit, left_rc)
        self.updateLine(self.rightLine, right_fitx, right_fit, right_rc)

        return self.leftLine.bestx, self.rightLine.bestx, ploty

class AdvancedLaneDetection ():
    def __init__(self):
        self.cameraCalibration = CameraCalibration('camera_cal/', CHESSBOARD_SIZE)

        self.colorGradientSpace = ColorGradientSpace(ABSOLUTE_GRADIENT_THRESHOLD, MAGNITUDE_GRADIENT_THRESHOLD,
                    DIRECTION_GRADIENT_THRESHOLD, COLOR_CHANNEL_THRESHOLD, SOBEL_KERNEL_SIZE, COLOR_CHANNEL)
                    
        self.perspectiveTransform = PerspectiveTransform(PERSPECTIVE_SRC, PERSPECTIVE_DST)
        
        self.laneDetect = LaneDetect(NUMBER_OF_WINDOWS, WINDOW_MARGIN, MINIMUM_PIXEL_IN_WINDOW, METER_PER_PIXEL_X, METER_PER_PIXEL_Y, N_FRAMES)

    def pipeline (self, image):


        # undist_image = cameraCalibration.undistortImage(image)
        # print('Camera Calibration completed!')

        undist_image = image
        warped_image, M_inv = self.perspectiveTransform.warped(
                        self.colorGradientSpace.colorGradientCombined(undist_image))
        # print('Warped, and color and Gradient space completed!')

        left_fitx, right_fitx, ploty = self.laneDetect.detectLine(warped_image)

        # print('Lane detected!')
        output_image = utils.drawOnResults(image, undist_image, warped_image,
                            left_fitx, right_fitx, ploty, M_inv)
                            
        return output_image

utils = Utils()

# step 1 camera calibration
# cameraCalibration = CameraCalibration('camera_cal/', CHESSBOARD_SIZE)
# utils.applyFilter('camera_cal/straight_lines1.', 'calibrated_test_images/', cameraCalibration.undistortImage)

# # step2 color and gradient space
# colorGradientSpace = ColorGradientSpace(ABSOLUTE_GRADIENT_THRESHOLD, MAGNITUDE_GRADIENT_THRESHOLD,
#                     DIRECTION_GRADIENT_THRESHOLD, COLOR_CHANNEL_THRESHOLD, SOBEL_KERNEL_SIZE, COLOR_CHANNEL, COLOR_CODE)
# utils.applyFilter('calibrated_test_images/', 'gradient_test_images/', colorGradientSpace.colorGradientCombined)

# # step 3 perspective transform
perspectiveTransform = PerspectiveTransform(PERSPECTIVE_SRC, PERSPECTIVE_DST)
utils.applyFilter('gradient_test_images/straight_lines', 'warped_test_images/', perspectiveTransform.warped)

# # step 4 detect lane and radius of curvature
laneDetect = LaneDetect(NUMBER_OF_WINDOWS, WINDOW_MARGIN, MINIMUM_PIXEL_IN_WINDOW, METER_PER_PIXEL_X, METER_PER_PIXEL_Y, N_FRAMES)
utils.applyFilter('warped_test_images/straight_lines', 'laneDetected_test_images/', laneDetect.fit_polynomial)

# advancedLaneDetection = AdvancedLaneDetection()

# project_video_output_fname = 'project_video_output.mp4'
# project_video = VideoFileClip("project_video_calibrated.mp4")

# project_video_output = project_video.fl_image(advancedLaneDetection.pipeline)
# project_video_output.write_videofile(project_video_output_fname, audio=False)
# utils.applyFilter('test_images/test6', 'final_test_images/', advancedLaneDetection.pipeline)

# image = cv2.imread('test_images/straight_lines1.jpg')
# plt.plot(520, 500, '.')
# plt.plot(225, 700, '.')
# plt.plot(1075, 700, '.')
# plt.plot(760, 500, '.')
# plt.imshow(image)
# plt.show()

image = cv2.imread('test_images/straight_lines1.jpg')

utils.showImages(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(perspectiveTransform.warped(image), cv2.COLOR_BGR2RGB))