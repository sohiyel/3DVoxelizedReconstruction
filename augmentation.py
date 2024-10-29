import cv2 as cv
import numpy as np
import os

def draw_axes(img, mtx, dist, rVec, tVec):
    length = 6 * 115
    axes = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,-length]])
    imgpts, _ = cv.projectPoints(axes, rVec, tVec, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1,2)
    #draw axis
    cv.line(img, imgpts[0], imgpts[1], (255,0,0), 2)
    cv.line(img, imgpts[0], imgpts[2], (0,255,0), 2)
    cv.line(img, imgpts[0], imgpts[3], (0,0,255), 2)
    return img

def draw_cube(img, mtx, dist, objPoint, corners):
    """ Convert vectors to image points and 
    Draw a cube and 3D world axes on the origin of the world"""
    axis = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0],
                    [0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2], [3,0,0],[0,3,0],[0,0,-3] ])

    _,rvecs, tvecs = cv.solvePnP(objPoint, corners, mtx, dist)
    imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    # draw cube
    add_cube(img, imgpts)
    return img

def add_cube(img, imgpts):
    """ Draw a cube and 3D world axes based on the image points"""
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw floor
    cv.drawContours(img, [imgpts[:4]],-1,(100,150,10),-1)

    # draw pillars
    for i,j in zip(range(4),range(4,8)):
        cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),1)

    # draw roof
    cv.drawContours(img, [imgpts[4:8]],-1,(200,150,10),1)

def add_line(img, pt0, pt1):
    cv.line(img, pt0, pt1, (255,0,0), 1)
    return img

def add_voxel(img, center, length, rvecs, tvecs, mtx, dist):
    halfLength = length/2
    axis = np.float32([
        [center[0]-halfLength,center[1]-halfLength,center[2]-halfLength],
        [center[0]-halfLength,center[1]+halfLength,center[2]-halfLength],
        [center[0]+halfLength,center[1]+halfLength,center[2]-halfLength],
        [center[0]+halfLength,center[1]-halfLength,center[2]-halfLength],
        [center[0]-halfLength,center[1]-halfLength,center[2]+halfLength],
        [center[0]-halfLength,center[1]+halfLength,center[2]+halfLength],
        [center[0]+halfLength,center[1]+halfLength,center[2]+halfLength],
        [center[0]+halfLength,center[1]-halfLength,center[2]+halfLength]
        ])

    imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    # draw cube
    add_cube(img, imgpts)
    return img
