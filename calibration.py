import cv2 as cv
import numpy as np
import os
from external_modules import *
from augmentation import *

class Calibration:
    def __init__(self,
                 camNum,
                 voxelSize,
                 dataPath = r"data",
                 imgPath = r"images",
                 drawCornerPath = r"draw_corners",
                 cubesPath = r"cubes",
                 intPath = r"intrinsics.yml",
                 extPath = r"extrinsics.yml") -> None:
        self.camMtx = ""
        self.distMtx = ""
        self.rVec = ""
        self.tVec = ""
        self.uOfBoard = 6
        self.vOfBoard = 8
        self.dataPath = os.path.join(os.path.abspath( os.path.curdir), dataPath)
        self.imgPath = os.path.join(self.dataPath, imgPath)
        self.drawCornerPath = os.path.join(self.dataPath, drawCornerPath)
        self.intPath = intPath
        self.extPath = extPath
        self.cubesPath = os.path.join(self.dataPath, cubesPath)
        self.camPath = os.path.join(self.dataPath,"cam"+str(camNum))
        if not os.path.exists(self.imgPath):
            os.mkdir (self.imgPath)
        if not os.path.exists(self.drawCornerPath):
            os.mkdir(self.drawCornerPath)
        if not os.path.exists(self.cubesPath):
            os.mkdir(self.cubesPath)
        self.objPoint = np.zeros((self.uOfBoard*self.vOfBoard,3), np.float32)
        self.objPoint[:,:2] = (voxelSize*np.mgrid[0:self.uOfBoard,0:self.vOfBoard]).T.reshape(-1,2)

    def find_corners(self, img, online = True):
        """ Find the corners of the chess board
        -First tries to do it with openCV function
        -If it didn't work, it tries to give it from user
        -The order of the corner points are:
        Buttum-Right -> Buttum-Left -> Top-Right -> Top-Left"""
        CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gImg, (self.uOfBoard,self.vOfBoard), None)
        if not ret and online:
            return False, ""

        if  not ret and not online:
            cv.imshow("image", img)

            corner_points = []
            def manual_annotation(event, x, y, flags, params):
                """Asks to manually give the coordinates of the chessboard corners.
                The 4 corners should be given from left to right and from top to bottom."""

                if event == cv.EVENT_LBUTTONDOWN:
                    # display the coordinates on the shell
                    print(str(x) + ", " + str(y))
                    corner_points.append([x, y])
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img, str(x) + ", " + str(y),
                               (x, y), font, 0.25, (0, 0, 255), 1)
                    cv.imshow("image", img)

            cv.setMouseCallback("image", manual_annotation)
            cv.waitKey(0)
            corner_points = np.array(corner_points, np.float32)
            if corner_points.shape == (4, 2):
                # ret, corners = interpolate(corner_points)
                width, height = self.uOfBoard, self.vOfBoard
                transform_points = np.array([[width - 1, 0], [0, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
                perspective_matrix = cv.getPerspectiveTransform(np.array(corner_points, dtype=np.float32), transform_points)
                inverse_perspective_matrix = np.linalg.pinv(perspective_matrix)

                # interpolate chessboard corners, 
                grid_coordinates = np.zeros((width * height, 2), np.float32)
                grid_coordinates[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
                corners = cv.perspectiveTransform(np.array([[grid_point] for grid_point in grid_coordinates], np.float32), inverse_perspective_matrix)
                cv.destroyAllWindows()

        # corners2 = cv.cornerSubPix(gImg,corners, (11,11), (-1,-1), CRITERIA)
        return True, corners

    def find_all_corners(self, inputPath, draw = False):
        """ Get Corner points from the find_corner function for all images
        in the database and return the objPoints and imgPoints.
        If draw parameter set to True, it will save the images with drawn corners"""
        objPoints = [] # 3d point in real world space
        imgPoints = [] # 2d points in image plane.
        
        cam = inputPath.split("\\")[-1]
        outPath = os.path.join(self.drawCornerPath,cam)

        if not os.path.exists(outPath):
            os.mkdir (outPath)

        for i in os.listdir(inputPath):
            img = cv.imread(os.path.join(inputPath, i), 1)
            gImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = resize_images(img)
            ret, corners = self.find_corners(img, False)
            if ret:
                imgPoints.append(corners)
                objPoints.append(self.objPoint)
                if draw:
                    img = cv.drawChessboardCorners(img, (self.uOfBoard, self.vOfBoard), corners, False)
                    filePath = os.path.join(outPath, i)
                    cv.imwrite(filePath, img)

        return objPoints, imgPoints, gImg

    def calibrate_camera_intrinsics(self):
        """Estimates the coeffiecients for camera calibration and saves them in a csv file
        - camMtx: the intrinsic camera matrix [3x3]
        - distMtx: lens distortion coefficients [1x5]
        - rotvecs: the rotation matrix [3x1] for each image
        - tvecs: the translation matrix [3x1] for each image"""

        coefPath = os.path.join(self.camPath,self.intPath)
        if os.path.exists(coefPath):
            return self.load_coefficients(True, coefPath)
        else:
            imPath = os.path.join( self.imgPath, "cam"+self.camPath[-1])
            if not os.path.exists(imPath):
                extract_frames_from_video(self.imgPath,self.camPath,"intrinsics.avi",self.uOfBoard,self.vOfBoard,True,25)
            objPoints, imgPoints, gImages = self.find_all_corners(imPath, True)
            ret, self.camMtx, self.distMtx, rotVecs, transVecs = cv.calibrateCamera(objPoints, imgPoints, gImages.shape[::-1], None, None)
            self.save_coefficients(self.camMtx, self.distMtx, True, coefPath)
            return [self.camMtx, self.distMtx]

    def draw_cubes(self):
        """ This function draw a cube on the origin of the world space
        on every images on the database"""
        mtx, dist = self.calibrate_camera()
        for i in os.listdir(self.imgPath):
            img = cv.imread(os.path.join(self.imgPath, i), 1)
            img = resize_images(img)
            ret, corners = self.find_corners(img)
            img = draw_cube(img, mtx, dist, self.objPoint, corners)
            filePath = os.path.join(self.cubesPath, i)
            cv.imwrite(filePath, img)

    def save_coefficients(self, mtx, dist, intrinsics=True, path="coef.csv"):
        """Save the camera matrix and the distortion coefficients to given path/file."""
        cvFile = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
        if intrinsics:
            cvFile.write('K', mtx)
            cvFile.write('D', dist)
        else:
            cvFile.write('R', mtx)
            cvFile.write('T', dist)
        # note you *release* you don't close() a FileStorage object
        cvFile.release()


    def load_coefficients(self, intrinsics=True, path="coef.csv"):
        """Loads camera matrix and distortion coefficients."""

        # FILE_STORAGE_READ
        cvFile = cv.FileStorage(path, cv.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve otherwise we only get a
        # FileNode object back instead of a matrix
        if intrinsics:
            self.camMtx = cvFile.getNode('K').mat()
            self.distMtx = cvFile.getNode('D').mat()
            cvFile.release()
            return [self.camMtx, self.distMtx] 
        else:
            self.rVec = cvFile.getNode('R').mat()
            self.tVec = cvFile.getNode('T').mat()
            cvFile.release()
            return [self.rVec, self.tVec] 


    def calibrate_camera_extrinsics(self):
        """Calibrate camera extrinsics"""
        exPath = os.path.join(self.camPath,self.extPath)
        if os.path.exists(exPath):
            return self.load_coefficients(False, exPath)
        else:
            mtx, dist = self.load_coefficients(True,os.path.join(self.camPath, self.intPath))
            imPath = os.path.join( self.imgPath, "cam"+self.camPath[-1])
            extract_frames_from_video(self.imgPath,self.camPath,"checkerboard.avi",self.uOfBoard,self.vOfBoard,False, 1)
            for j in os.listdir(imPath):
                if j.split("-")[0] == "checkerboard":
                    img = cv.imread(os.path.join(imPath, j), 1)
                    gImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    img = resize_images(img)
                    ret, corners = self.find_corners(img, False)
                    if ret:
                        _,rvecs, tvecs = cv.solvePnP(self.objPoint, corners, mtx, dist)
                        self.save_coefficients(rvecs, tvecs, False, exPath)
                        img = draw_axes(img, mtx, dist, rvecs, tvecs)
                        outPath = os.path.join(self.cubesPath, self.camPath[-1]+"-"+j)
                        cv.imwrite(outPath,img)
                        return [rvecs, tvecs]