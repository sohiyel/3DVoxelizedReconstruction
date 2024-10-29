import numpy as np
import os
import cv2 as cv

def interpolate(corners: np.ndarray):
    """Interpolates the cell points from the chessboard corners."""
    ret = True
    all_points = []
    horizontal_up = get_equidistant_points(corners[1], corners[0], 7)
    horizontal_down = get_equidistant_points(corners[3], corners[2], 7)
    for j in range(len(horizontal_up)):
        all_points.append([list(p) for p in get_equidistant_points(horizontal_up[j], horizontal_down[j], 5)])
    corners = np.array(all_points, np.float32).reshape(48, 1, 2)
    return ret, corners

def get_equidistant_points(p1, p2, n):
    """Gets n points evenly distanced between point1 and point2"""
    return list(zip(np.linspace(p1[0], p2[0], n+1),
                np.linspace(p1[1], p2[1], n+1)))


def resize_images(img, baseWidth = 1080):
    return img
    """The initial chessboard images we took were very large,
    so we created this function to reduce their size."""
    height, width = img.shape[:2]
    wPercent = (baseWidth / width)
    hSize = int((height * float(wPercent)))
    dim = (baseWidth, hSize)
    
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized

def extract_frames_from_video(imgPath, camPath, videoName, uOfBoard=6, vOfBoard=8, checkCorner=True, count=1000, frameRate = 50):
        imPath = os.path.join( imgPath, "cam"+camPath[-1])
        if not os.path.exists(imPath):
            os.mkdir (imPath)
        # Opens the Video file
        cap= cv.VideoCapture(os.path.join(camPath, videoName))
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False or i > (count*frameRate):
                break
            if i% (frameRate) == 0:
                ret, corners = cv.findChessboardCorners(frame, (uOfBoard,vOfBoard), None)
                if ret or not checkCorner:
                    cv.imwrite(os.path.join(imPath,videoName.split(".")[0]+"-"+str(i)+'.jpg') ,frame)
            i+=1
        
        cap.release()
        cv.destroyAllWindows()