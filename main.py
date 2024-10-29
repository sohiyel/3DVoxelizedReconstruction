from calibration import Calibration
from camera import Camera
import sys
from concurrent.futures import ProcessPoolExecutor

def create_lookup(cam):
    cam.calibrate()
    cam.generate_lookup_table()

if __name__ == "__main__":
    cam1 = Camera(1)
    cam2 = Camera(2)
    cam3 = Camera(3)
    cam4 = Camera(4)
    cams = [cam1, cam2, cam3, cam4]
    if sys.argv[1] == "calibrate":
        cam1.calibrate()
        print("\n1->camMtx:",cam1.camMtx,"\ndistMtx:", cam1.distMtx,"\nrVec:", cam1.rVec,"\ntVec:", cam1.tVec)
        cam2.calibrate()
        print("\n2->camMtx:",cam2.camMtx,"\ndistMtx:", cam2.distMtx,"\nrVec:", cam2.rVec,"\ntVec:", cam2.tVec)
        cam3.calibrate()
        print("\n3->camMtx:",cam3.camMtx,"\ndistMtx:", cam3.distMtx,"\nrVec:", cam3.rVec,"\ntVec:", cam3.tVec)
        cam4.calibrate()
        print("\n4->camMtx:",cam4.camMtx,"\ndistMtx:", cam4.distMtx,"\nrVec:", cam4.rVec,"\ntVec:", cam4.tVec)
    if sys.argv[1] == "imgPoint":
        cam1.calibrate()
        print(cam1.get_img_point((0,0,0)))
    if sys.argv[1] == "lookup":
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     executor.map(create_lookup, cams)
        create_lookup(cam1)
        create_lookup(cam2)
        create_lookup(cam3)
        create_lookup(cam4)
    if sys.argv[1] == "load_lookUp":
        for cam in cams:
            cam.load_lookup_table()
            print(cam.lookUpTable)
            print(cam.lookUpTable.max())
            print(cam.lookUpTable.min())
    if sys.argv[1] == "show":
        for c in cams:
            c.calibrate()
            c.load_lookup_table()
            c.show_lookup_table()
        # cam1.calibrate()
        # cam1.load_lookup_table()
        # cam1.show_lookup_table()
    