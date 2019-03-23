import numpy as np
import cv2

"""
This class contain information of the line
"""
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

"""
This class detect lane from the image using sliding window technique
"""
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

    def find_lane_pixels_sliding_window(self, image):
        left_lane_inds = []
        right_lane_inds = []
        window_height = np.int(image.shape[0] // self.n_windows)
        
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)

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
        
        return leftx, lefty, rightx, righty
        
    def fit_polynomial (self, image, leftx, lefty, rightx, righty):

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def sliding_window (self, image):
        leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window (image)

        return self.fit_polynomial(image, leftx, lefty, rightx, righty)

    def search_from_prior(self, image):
        
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (self.leftLine.current_fit[0]*(nonzeroy**2) + self.leftLine.current_fit[1]*nonzeroy + 
                    self.leftLine.current_fit[2] - self.margin)) & (nonzerox < (self.leftLine.current_fit[0]*(nonzeroy**2) + 
                    self.leftLine.current_fit[1]*nonzeroy + self.leftLine.current_fit[2] + self.margin)))
        right_lane_inds = ((nonzerox > (self.rightLine.current_fit[0]*(nonzeroy**2) + self.rightLine.current_fit[1]*nonzeroy + 
                    self.rightLine.current_fit[2] - self.margin)) & (nonzerox < (self.rightLine.current_fit[0]*(nonzeroy**2) + 
                    self.rightLine.current_fit[1]*nonzeroy + self.rightLine.current_fit[2] + self.margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        #if lane pixel found using search prior are less than 80% of total pixels we run sliding window
        if (len(leftx) + len(rightx)) < len(nonzerox)*0.8:
            print('Searched from prior doesn\'t work searching using sliding window!')   
            return self.sliding_window(image)
        
        self.leftLine.allx = leftx
        self.leftLine.ally = lefty
        self.rightLine.allx = rightx
        self.rightLine.ally = righty

        return self.fit_polynomial(image, leftx, lefty, rightx, righty)

    def get_curvature (self, ploty):
        
        y_eval = np.max(ploty)
        left_fit_real = np.polyfit(self.leftLine.ally*self.ym_per_pixel, self.leftLine.allx*self.xm_per_pixel, 2)
        right_fit_real = np.polyfit(self.rightLine.ally*self.ym_per_pixel, self.rightLine.allx*self.xm_per_pixel, 2)

        left_curverad = ((1 + (2*left_fit_real[0]*y_eval*self.ym_per_pixel + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
        right_curverad = ((1 + (2*right_fit_real[0]*y_eval*self.ym_per_pixel + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])

        return left_curverad, right_curverad

    def get_vehicle_position_text (self):
        return '{0:.2f} m towards left from centre'.format(self.leftLine.line_base_pos - self.rightLine.line_base_pos)

    def updateLine (self, line, fitx, fit, radius_of_curvature):
        
        last_fit = None
        if line.detected:
            last_fit = line.current_fit

        line.current_fit = fit

        if line.detected:
            line.diffs = last_fit - line.current_fit

        line.detected = True
        
        line.recent_fits.append(line.current_fit)
        line.recent_xfitted.append(fitx)
        if len(line.recent_xfitted) > self.n_frames:
            line.recent_xfitted = line.recent_xfitted[1:]
            line.recent_fits = line.recent_fits[1:]

        line.bestx = np.mean(line.recent_xfitted, axis=0)
        line.best_fit = np.mean(line.recent_fits, axis=0)
        line.radius_of_curvature = radius_of_curvature
        
    def detectLine (self, image):

        if self.leftLine.detected and self.rightLine.detected:
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.search_from_prior(image)
        else:
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.sliding_window (image)
        
        left_rc, right_rc = self.get_curvature(ploty)

        #update lane lines position with respect to center
        self.leftLine.line_base_pos = (image.shape[1]//2 - np.mean(self.leftLine.allx))*self.xm_per_pixel
        self.rightLine.line_base_pos = (np.mean(self.rightLine.allx) - image.shape[1]//2)*self.xm_per_pixel

        #update lane line information
        self.updateLine(self.leftLine, left_fitx, left_fit, left_rc)
        self.updateLine(self.rightLine, right_fitx, right_fit, right_rc)

        return self.leftLine.bestx, self.rightLine.bestx, ploty
