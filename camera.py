import os
from calibration import Calibration
import cv2 as cv
import numpy as np
import pandas as pd
from external_modules import resize_images
from augmentation import add_cube, add_line, add_voxel
from engine.config import config
import random
from background_substraction import BackgroundSubstraction
class Camera:
    def __init__(self,
                 camNum,
                 dataPath = r"data") -> None:
        self.camNum = camNum
        self.camMtx = ""
        self.distMtx = ""
        self.rVec = ""
        self.tVec = ""
        self.lookUpTable = pd.DataFrame(columns=["imgPtU", "imgPtV", "wPtX", "wPtY", "wPtZ", "active"])
        self.lookUpSize = (config["lookup_width"],config["lookup_depth"],config["lookup_height"])
        self.voxelSize = 115
        self.calibration = Calibration(self.camNum,self.voxelSize,dataPath)
        self.backgroundSubstractor = BackgroundSubstraction(self.calibration.camPath)
        self.dataPath = os.path.join(os.path.abspath( os.path.curdir), dataPath)

    def calibrate(self):
        """Calibrate and set the camera matrices"""
        self.camMtx, self.distMtx = self.calibration.calibrate_camera_intrinsics()
        self.rVec, self.tVec = self.calibration.calibrate_camera_extrinsics()

    def get_img_point(self, wPoint):
        """Convert world space points to image points"""
        imgpts, _ = cv.projectPoints(np.float32([wPoint]), self.rVec, self.tVec, self.camMtx, self.distMtx)
        return imgpts[0]
    
    def generate_lookup_table(self):
        """Generate a lookup table consists of some world space, image points,
          and the status of that point whether it should be active or not and
          save them to a file under camera folder"""
        for x in range(self.lookUpSize[0]):
            for y in range(self.lookUpSize[1]):
                for z in range(self.lookUpSize[2]):
                    wX = self.voxelSize*(x - self.lookUpSize[0]/2)
                    wY = self.voxelSize*(y - self.lookUpSize[2]/2)
                    wZ = -1*self.voxelSize*z
                    wPt = (wX, wY, wZ)
                    imgPt = self.get_img_point(wPt)
                    # if imgPt[0][0] > 644 or imgPt[0][1] > 486 or imgPt[0][0] < 0 or imgPt[0][1] < 0:
                    #     continue
                    active = True
                    newRow = pd.Series({"imgPtU": int(imgPt[0][0]), "imgPtV":int(imgPt[0][1]), "wPtX":wX, "wPtY":wY, "wPtZ":wZ, "active":active})
                    self.lookUpTable = pd.concat(
                        [
                            self.lookUpTable,  
                            newRow.to_frame().T
                        ],
                        ignore_index=True)
        self.lookUpTable.to_csv(os.path.join(self.calibration.camPath,"lookUpTable.csv"))

    def load_lookup_table(self):
        """Load generated lookup table"""
        self.lookUpTable = pd.read_csv(os.path.join(self.calibration.camPath,"lookUpTable.csv"))
                
    def show_lookup_table(self):
        """Create a test image on a frame of the chessboard video
            that shows the location of size of the lookup table and its voxels"""
        img = cv.imread(os.path.join(self.calibration.imgPath,"cam"+str(self.camNum),"checkerboard-0.jpg"), 1)
        img = resize_images(img)
        for index, row in self.lookUpTable.iterrows():
            if row["active"]:
                add_voxel(img, [row["wPtX"],row["wPtY"],row["wPtZ"]], self.voxelSize, self.rVec, self.tVec, self.camMtx, self.distMtx)
        cv.imwrite("test"+str(self.camNum)+".png", img)