## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Undistorted"
[image2]: ./output_images/undistorted_image.jpg "Road Transformed"
[image3]: ./output_images/gradient_image.jpg "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/lane_detected_image.jpg "Fit Visual"
[image6]: ./output_images/final_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `code/calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objectp` is just a replicated array of coordinates, and `objectpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imagepoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objectpoints` and `imagepoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `code/color_and_gradient_thresholding.py`).

I start by calculating the absolute sobel gradient in both x and y directions with a threshold. Similary, I calculated the magnitude gradient and direction gradient of the image. Then, I tried the various combinations of these gradient to get the best result. Finally, I end up with the following condition in gradient threshold

```python
combined[((gradx == 255) & (grady == 255)) | ((mag_binary == 255) & (dir_binary == 255))] = 255
```

Then I color transformed the images in S channel with thresholding. I also tried the other color channels but S (or saturated) gave me the best results that's why I use it. 

Finally, I combined the color threshold and gradient threshold binary images to get a combined thresholded binary image, and get the following result.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()` in the `BirdEye` class, which appears in the file `code/bird_eye.py`.  The `BirdEye` class takes as source (`src`) and destination (`dst`) points, and then use them in `warper()` function which taked (`img`) as an image.  I selected the source points from the image, and then set destination points accordingly. 

Here is my source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 205, 720      | 320, 720      |
| 1110, 720     | 960, 720      |
| 703, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In this step, I used sliding window technique to detect the left and right lane pixel. The code of sliding window is in lines #46 through #98 in `code\lane_detection.py`. From then, I simply compute a second degree polynomial, via numpy’s polyfit, to find the coefficients of the curves that best fit the left and right lane lines.

To get the starting x position of the lanes, I computed the histogram on the bottom half of the binary threshold warped image which I got from previous step, and then identify the x positions where the pixel intensities are highest.

To improve the algorithm, I used `search_from_prior()` function implmented in lines #117 through #144 of the file `code\lane_detection.py`. In this function, I used the previously computed polynomial cofficient to get the lane pixels x and y position in the +/- margin of the old line center. However, when I do not find enough lane line pixels (less than 80% of total non zero pixels), I revert to sliding windows search to help improve our chances of fitting better curves around our lane.

To make the cleaner result, I also do smoothing. Each time I get a new high-confidence measurement, I append it to the list of recent measurements and then take an average over `n_frames` past measurements to obtain the lane position I want to draw onto the image.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in lines #146 through #155 and the position of vehicle with respect to center in lines #193 through #194 in my code in `code/lane_detection.py`. As, I calculated the fitted polynomial coeffecient in the previous step, now I can use these coefficient to calculate the radius of curvature of the lane.

I used the following formula to calculate the radius of curvature where A, B, and C are the polynomial coefficient.

$R_{curve}$ = $\frac{(1+(2Ay+B)^2)^3/_2}{|2A|}$

I also used meter per pixel in x and y direction of image to convert the radius of curvature from image domain to the real world.

T0 calculate the vehicle position from center I substracted the vehicle distance from right lane from vehicle distance from left lane.

vehicle position from left = distance from left lane - distance from right lane


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #53 through #84 in my code in `code/pipeline.py` in the function `map_lane()` and `put_text()`. The `put_text()` function put text on the original image like radius of curvature and vehicle position. The `map_lane()` function mapped the lane on original image.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I really found it difficult to get the right parameters for gradient thresholding which is very important to get a curve that makes sense. I think I still need to tweak to get a better radius of curvature estimates. I also think I could make a more clean readable implementation of the sliding window algorithm to get the pixels of interest for each lane line if I give it more thought.

My algorithm also don't give the best results when the light is too high. So, I am working on the gradient and color thresholding to cover the all light conditions. It also failed when the lane comes out from the edge of image maybe because I haven't handled the outliers. Radius of curvature calculated is also not too good.

I am focusing to work on the above issues, and will try improve it.