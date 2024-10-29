from external_modules import extract_frames_from_video
import os
import cv2 as cv
import skimage.filters
import matplotlib.pyplot as plt
import numpy as np
import random

class BackgroundSubstraction:
    def __init__(self, camPath) -> None:
        self.backgroundModel = ""
        self.backgroundPath = os.path.join(camPath, "background.avi")
        self.bgMedian = ""
        self.train_background_model()
        self.find_the_median()

    def train_background_model(self):
        # train MOG2 on background video, remove shadows, default learning rate
        self.backgroundModel = cv.createBackgroundSubtractorMOG2()
        self.backgroundModel.setShadowValue(0)

        # open background.avi
        camera_handle = cv.VideoCapture(self.backgroundPath)
        num_frames = int(camera_handle.get(cv.CAP_PROP_FRAME_COUNT))

        # train background model on each frame
        for i_frame in range(num_frames):
            ret, image = camera_handle.read()
            if ret:
                self.backgroundModel.apply(image)

        # close background.avi
        camera_handle.release()

    # applies background subtraction to obtain foreground mask
    def background_subtraction(self, image):
        foreground_image = self.backgroundModel.apply(image, learningRate=0)

        # remove noise through dilation and erosion
        erosion_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        dilation_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        foreground_image = cv.dilate(foreground_image, dilation_elt)
        foreground_image = cv.erode(foreground_image, erosion_elt)
        foreground_image = cv.erode(foreground_image, erosion_elt)
        foreground_image = cv.erode(foreground_image, erosion_elt)
        foreground_image = cv.dilate(foreground_image, dilation_elt)

        
                
        return foreground_image

    def find_the_median(self):
        video = cv.VideoCapture(self.backgroundPath)
        """Find the median of some random sample from the video in the HSV colorspace"""
        totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)
        sample = int(totalFrames * 0.2)
        randomFrameNumbers = []
        for j in range(sample):
            randomFrameNumbers.append(random.randint(0, totalFrames))

        frames = []
        for randomFrame in randomFrameNumbers:
            video.set(cv.CAP_PROP_POS_FRAMES, randomFrame)
            success, image = video.read()
            if success:
                hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                frames.append(hsvImage)

        medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
        self.bgMedian = medianFrame


# def background_substraction (frame, backGroundPath):
#     """ Background substraction algorithm that takes a frame and the path 
#         to the background refrence as input and takes the foreground mask as output"""
#     # Create background substraction algorithm

#     # Capture the video from the webcam or from the filepath
#     captureBackground = cv.VideoCapture(backGroundPath)

#     if not captureBackground.isOpened():
#         print("Unable to open or play the video")
#         exit(0)

#     medianFrame = find_the_median(captureBackground)

#     hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     # Calculate absolute difference of current frame and
#     # the median frame
#     foregroundMask = cv.absdiff(hsvFrame, medianFrame)
#     # Treshold to binarize
#     # Setting the threshold automatically?
#     th, foregroundMask = cv.threshold(foregroundMask, 50, 250, cv.THRESH_BINARY)
#     foregroundMask = cv.cvtColor(foregroundMask, cv.COLOR_BGR2GRAY)
#     th, foregroundMask = cv.threshold(foregroundMask, 70, 200, cv.THRESH_BINARY)
#     # Post-processing of the frame to reduce noisy pixels that represent the background
#     # Create empty kernel for the dilution functions
#     kernel = np.ones((3, 3), np.uint8)
#     foregroundMask = cv.dilate(foregroundMask, kernel, iterations=2)

#     # function that will give back the foregroundmask and backgroundmask seperatly
#     return foregroundMask

    