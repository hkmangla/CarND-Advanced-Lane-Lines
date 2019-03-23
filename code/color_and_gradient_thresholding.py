import numpy as np
import cv2

"""
Below class compute color and gradient threshold of the image
"""
class ColorGradientSpace:
    def __init__ (self, absGradThreshold, magGradThreshold, dirGradThreshold, colorThreshold=(0,255), kernel_size=3, colorChannel='S'):
        self.absGradThreshold = absGradThreshold
        self.magGradThreshold = magGradThreshold
        self.dirGradThreshold = dirGradThreshold
        self.colorThreshold = colorThreshold
        colorChannelDict = {'H': 0, 'L': 1, 'S': 2}
        self.colorChannel = colorChannelDict[colorChannel]
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
    
    def HLSChannel (self, image):
        HLSImage = cv2.cvtColor (image, cv2.COLOR_BGR2HLS)

        channel_image = HLSImage[:, :, self.colorChannel]
        binary = np.zeros_like(channel_image)
        binary[(channel_image <= self.colorThreshold[1]) & (channel_image >= self.colorThreshold[0])] = 255

        return binary
        
    def gradientCombined (self, image):

        gradx = self.absoluteGradient(image, orient='x', thresh=self.absGradThreshold)
        grady = self.absoluteGradient(image, orient='y', thresh=self.absGradThreshold)
        mag_binary = self.magnitudeGradient(image, thresh=self.magGradThreshold)
        dir_binary = self.directionGradient(image, thresh=self.dirGradThreshold)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 255) & (grady == 255)) | ((mag_binary == 255) & (dir_binary == 255))] = 255

        return combined

    def colorGradientCombined (self, image):
        #get colore threshold binary image
        colored = self.HLSChannel(image)

        #get gradient threshold binary image
        gradientCombined = self.gradientCombined(image)

        #combined the color and gradient threshold images to get the best result
        combined = np.zeros_like(gradientCombined)
        combined[(colored == 255) | (gradientCombined == 255)] = 255

        return combined