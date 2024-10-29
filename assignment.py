import glm
import random
import numpy as np
from camera import Camera
import cv2 as cv
import os
from external_modules import resize_images
from concurrent.futures import ProcessPoolExecutor

def generate_lookups():
    for c in cams:
        c.generate_lookup_table()

def load_lookups():
    for c in cams:
        c.load_lookup_table()

def calibrate():
    for c in cams:
        c.calibrate()

block_size = 1.0
cam1 = Camera(1)
cam2 = Camera(2)
cam3 = Camera(3)
cam4 = Camera(4)
cams = [cam1, cam2, cam3, cam4]
calibrate()
load_lookups()

block_size = 1.0

def check_voxels():
    """ For each camera, takes the background substraction mask
        Then, update the lookup table for that camera based on the mask"""
    for i,c in enumerate(cams):
        camPath = c.calibration.camPath
        backGroundPath = os.path.join(camPath, "background.avi")
        # Load the video for which the background needs to be substracted
        captureVideos = cv.VideoCapture(os.path.join(camPath,"video.avi"))
        # Reset frame number to 0
        captureVideos.set(cv.CAP_PROP_POS_FRAMES, 0)

        ret, frameInputOutput = captureVideos.read()
        # mask = background_substraction(frameInputOutput, backGroundPath)
        mask = c.backgroundSubstractor.background_subtraction(frameInputOutput)
        mask = resize_images(mask)
        # cv.imshow('mask', mask)
        # cv.waitKey(0)
        rows, cols = mask.shape
        # print(rows, cols)

        for index, row in c.lookUpTable.iterrows():
            imgPtU = int(row["imgPtU"])
            imgPtV = int(row["imgPtV"])
            if imgPtV > rows-1 or imgPtU > cols-1 or imgPtV < 0 or imgPtU < 0:
                continue
            if mask[imgPtV,imgPtU] > 0:
                cams[i].lookUpTable.at[index,"active"] = True
            else:
                cams[i].lookUpTable.at[index,"active"] = False
    print("Checking voxels has been done")
    return cams

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    """ Check the lookup table of all cameras and if one voxel
        in the world space was active in all of them, it will be added
        to the scene"""
    data, colors = [], []
    cams = check_voxels()    
    for index, row in cams[0].lookUpTable.iterrows():
        if row["active"]:
            if cams[1].lookUpTable.iloc[index]["active"]:
                if cams[2].lookUpTable.iloc[index]["active"]:
                    if cams[3].lookUpTable.iloc[index]["active"]:
                        data.append([row["wPtX"]/cams[0].voxelSize, abs(row["wPtZ"]/cams[0].voxelSize), row["wPtY"]/cams[0].voxelSize])
                        colors.append([row["wPtX"] / (width * cams[0].voxelSize), row["wPtZ"] / (depth * cams[0].voxelSize), row["wPtY"] / (height * cams[0].voxelSize)])
    return data, colors


def get_cam_positions():
    """ Update camera positions based on the camera matrices"""
    camPoses = []
    for c in cams:
        R,_ = cv.Rodrigues(c.rVec)
        worldPos = -np.matrix(R).T * np.matrix(c.tVec) / cams[0].voxelSize
        y = abs(worldPos[2])
        worldPos[2] = worldPos[1]
        worldPos[1] = y
        camPoses.append(worldPos)
    
    return camPoses,[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    """ Update camera rotation based on the camera matrices"""
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
        
    for c in range(len(cam_rotations)):
        R,_ = cv.Rodrigues(cams[c].rVec)
        R4 = np.identity(4)
        R4[:3,:3] = R
        cam_rotations[c] = glm.mat4(R4)
        cam_rotations[c] = glm.rotate(cam_rotations[c], glm.radians(90), (0,0,1))
    return cam_rotations
