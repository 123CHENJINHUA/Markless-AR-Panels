"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math


def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 将每行分割成x, y, z三个部分并转换为浮点数
            x, y, z = map(float, line.strip().split())
            # 将坐标元组添加到列表中
            points.append(np.array([[x],[y],[z]],dtype = np.float64))
    return points

def Transformation(rvec_, tvec_,point):
    T_mask2cam = np.array([[tvec_[0][0]],[tvec_[1][0]],[tvec_[2][0]]],np.float64)
    R_mask2cam = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec_,R_mask2cam)

    Ptrans = np.dot(R_mask2cam,point)+T_mask2cam
    return Ptrans


def Point_Transform(points):
    
    width = 1920
    height = 1080
    fps = 60

    mark_size = 0.026

    cap = cv2.VideoCapture(1)

    cap.set(3, width)  #设置宽度
    cap.set(4, height)  #设置长度
    cap.set(5, fps)  
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


    cv_file = cv2.FileStorage("./charuco_camera_calibration.yaml", cv2.FILE_STORAGE_READ)

    # Note : we also have to specify the type
    # to retrieve otherwise we only get a 'None'
    # FileNode object back instead of a matrix
    mtx = cv_file.getNode("camera_matrix").mat()
    dist = cv_file.getNode("dist_coeff").mat()

    ###------------------ ARUCO TRACKER ---------------------------
    while (True):
        ret, frame = cap.read()

        if not ret:
            print("Unable to capture video")
            continue

        frame_copy = frame.copy()
        #if ret returns false, there is likely a problem with the webcam/camera.
        #In that case uncomment the below line, which will replace the empty frame 
        #with a test image used in the opencv docs for aruco at https://www.docs.opencv.org/4.5.3/singlemarkersoriginal.jpg
        # frame = cv2.imread('./images/test image.jpg') 

        # operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        board = cv2.aruco.CharucoBoard((3, 3), 0.035, 0.026, aruco_dict)
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 10

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):

            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, mark_size, mtx, dist)
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            # draw a square around the markers
            aruco.drawDetectedMarkers(frame_copy, corners)
            retval,charucoCorners,charucoIds =cv2.aruco.interpolateCornersCharuco(corners,ids,frame,board)
            if retval:
                retval, rvec_, tvec_ = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, None, None)

                # If pose estimation is successful, draw the axis
                if retval:

                    org_point = np.array([[0,0,0]],dtype = np.float64)
                    Porg, _ = cv2.projectPoints(org_point, rvec_, tvec_, mtx, dist)
                    Porg_x = int(Porg[0][0][0])
                    Porg_y = int(Porg[0][0][1])
                    cv2.circle(frame_copy,(Porg_x,Porg_y),10,(255,0,0),thickness=-1)
                    #cv2.drawFrameAxes(frame_copy, mtx, dist, rvec_, tvec_, length=0.05, thickness=2)

                    for point in points:
                        
                        Ptrans = Transformation(rvec_, tvec_,point)
                        Pdesign, _ = cv2.projectPoints(org_point, rvec_, Ptrans, mtx, dist)
                        Pdesign_x = int(Pdesign[0][0][0])
                        Pdesign_y = int(Pdesign[0][0][1])

                        if Pdesign_x < 0 or Pdesign_y < 0 or Pdesign_x > frame_copy.shape[1] or Pdesign_y > frame_copy.shape[0]:
                            continue

                        cv2.circle(frame_copy,(Pdesign_x,Pdesign_y),10,(0,0,255),thickness=-1)
                        #cv2.drawFrameAxes(frame_copy, mtx, dist, rvec_, P_c, length=0.05, thickness=2)

                else:
                    # code to show 'No Ids' when no markers are found
                    cv2.putText(frame_copy, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # display the resulting frame
        frame_copy = cv2.resize(frame_copy, (int(frame_copy.shape[1]/2), int(frame_copy.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('frame',frame_copy)
        k=cv2.waitKey(1)
        if k==27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    points_path = "points.txt"
    points = read_points(points_path)
    Point_Transform(points)


