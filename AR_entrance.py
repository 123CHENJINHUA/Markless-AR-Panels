from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import cv2.aruco as aruco
from PIL import Image
import numpy as np
import imutils
import sys

 
from tools.Visualize import draw_axis
from objloader import * #Load obj and corresponding material and textures.
from MatrixTransform import extrinsic2ModelView, intrinsic2Project 
from Filter import Filter

from Tracker.Tracking import Tracker


class AR_render:
    
    def __init__(self, camera_matrix, dist_coefs, object_path, building_path, model_scale = 0.03):
        
        """[Initialize]
        
        Arguments:
            camera_matrix {[np.array]} -- [your camera intrinsic matrix]
            dist_coefs {[np.array]} -- [your camera difference parameters]
            object_path {[string]} -- [your model path]
            model_scale {[float]} -- [your model scale size]
        """
        # Initialise webcam and start thread
        # self.webcam = cv2.VideoCapture(0)

        width = 1920
        height = 1080

        self.out_width = width//2
        self.out_height = height//2
        self.cam_matrix = camera_matrix
        self.dist_coefs = dist_coefs

        self.depth = 1.0

        fps = 60
        self.webcam = cv2.VideoCapture(0)

        self.webcam.set(3, width)  #设置宽度
        self.webcam.set(4, height)  #设置长度
        self.webcam.set(5, fps)  
        self.webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        self.tracker = Tracker(camera_matrix, dist_coefs, width, height, building_path)

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.tracker.click_and_crop)

        self.image_w, self.image_h = map(int, (self.webcam.get(3), self.webcam.get(4)))
        self.initOpengl(self.image_w, self.image_h)
        self.model_scale = model_scale
    
        self.cam_matrix,self.dist_coefs = camera_matrix, dist_coefs
        self.projectMatrix = intrinsic2Project(camera_matrix, self.image_w, self.image_h, 0.01, 100.0)
        self.loadModel(object_path)
        
        # Model translate that you can adjust by key board 'w', 's', 'a', 'd'
        self.translate_x, self.translate_y, self.translate_z = 0, 0, 0
        self.pre_extrinsicMatrix = None
        
        self.filter = Filter()
        

    def loadModel(self, object_path):
        
        """[loadModel from object_path]
        
        Arguments:
            object_path {[string]} -- [path of model]
        """
        self.model = OBJ(object_path, swapyz = True)

  
    def initOpengl(self, width, height, pos_x = 500, pos_y = 500, window_name = b'Aruco Demo'):
        
        """[Init opengl configuration]
        
        Arguments:
            width {[int]} -- [width of opengl viewport]
            height {[int]} -- [height of opengl viewport]
        
        Keyword Arguments:
            pos_x {int} -- [X cordinate of viewport] (default: {500})
            pos_y {int} -- [Y cordinate of viewport] (default: {500})
            window_name {bytes} -- [Window name] (default: {b'Aruco Demo'})
        """
        
        self.width = width
        self.height = height
        self.frames = []
        self.points_frames = []
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.out_width, self.out_height)
        glutInitWindowPosition(pos_x, pos_y)
           
        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        
        # # Assign texture
        glEnable(GL_TEXTURE_2D)
        
        # Add listener
        glutKeyboardFunc(self.keyBoardListener)
        
        # Set ambient lighting
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5,0.5,0.5,1)) 
        
        
        
        
 
    def draw_scene(self):
        """[Opengl render loop]
        """
        _, image = self.webcam.read()# get image from webcam camera.
        self.draw_background(image)  # draw background
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_objects(image, mark_size = 0.140) # draw the 3D objects.
        glutSwapBuffers()

        self.capture_frame()
        
        # TODO add close button
        # key = cv2.waitKey(20)
        
       
        
    def capture_frame(self):
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.out_width, self.out_height, GL_BGR, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.out_height, self.out_width, 3)
        image = cv2.flip(image, 0)  # OpenGL的坐标系和OpenCV不同，需要翻转
        self.frames.append(image)

    def save_video(self, filename='output.mp4', fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (self.out_width, self.out_height))
        for frame in self.frames:
            out.write(frame)
        out.release()

        filename2='output2.mp4'
        out2 = cv2.VideoWriter(filename2, fourcc, fps, (self.out_width, self.out_height))
        for frame in self.points_frames:
            out2.write(frame)
        out2.release()

    def keyBoardListener(self, key, x, y):
        if key == b'q':
            self.save_video()
            glutLeaveMainLoop()
 
 
    def draw_background(self, image):
        """[Draw the background and tranform to opengl format]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Setting background image project_matrix and model_matrix.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        fy = self.cam_matrix[1][1]
        fx = self.cam_matrix[0][0]
        fovy = 2*np.arctan(0.5*self.height/fy)*180/np.pi
        aspect = (self.width*fy)/(self.height*fx)

        gluPerspective(fovy, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
     
        # Convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

  
        # Create background texture
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
                
        ly = 2 * self.depth * np.tan(np.radians(fovy) / 2)/2
        lx = ly * aspect
        glTranslatef(0.0, 0.0, -self.depth)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-lx, -ly, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( lx, -ly, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( lx,  ly, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-lx,  ly, 0.0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
 

    def read_points(self, file_path):
        points = []
        with open(file_path, 'r') as file:
            for line in file:
                # 将每行分割成x, y, z三个部分并转换为浮点数
                x, y, z = map(float, line.strip().split())
                # 将坐标元组添加到列表中
                points.append(np.array([[x],[y],[z]],dtype = np.float64))
        return points
 
    def Transformation(self,rvec_, tvec_,point):
        T_mask2cam = np.array([[tvec_[0][0][0]],[tvec_[0][0][1]],[tvec_[0][0][2]]],np.float64)
        R_mask2cam = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec_,R_mask2cam)

        Ptrans = np.dot(R_mask2cam,point)+T_mask2cam
        return Ptrans

 
    def draw_objects(self, image, mark_size):
        """[draw models with opengl]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        
        Keyword Arguments:
            mark_size {float} -- [aruco mark size: unit is meter] (default: {0.07})
        """
        
        height,width,chanel = image.shape

        rvec, tvec, retval, image, _ = self.tracker.start_tracking(image)
        
        if retval:

            #cv2.drawFrameAxes(image,self.cam_matrix, self.dist_coefs, rvec, tvec, length=0.05, thickness=2)

            org_point = np.array([[0,0,0]],dtype = np.float64)
            points_path = "points.txt"
            points = self.read_points(points_path)
            for point in points:                        
                Ptrans = self.Transformation(rvec,tvec,point)
                Pdesign, _ = cv2.projectPoints(org_point, rvec, Ptrans, self.cam_matrix, self.dist_coefs)
                Pdesign_x = int(Pdesign[0][0][0])
                Pdesign_y = int(Pdesign[0][0][1])

                point_depth = Ptrans[2][0]*100
                if point_depth > self.depth:
                    self.depth = point_depth

                if Pdesign_x < 0 or Pdesign_y < 0 or Pdesign_x > image.shape[1] or Pdesign_y > image.shape[0]:
                    continue

                cv2.circle(image,(Pdesign_x,Pdesign_y),10,(0,0,255),thickness=-1)

            projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 1000.0)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glMultMatrixf(projectMatrix)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

    
            if tvec is not None:
                if self.filter.update(tvec): # the mark is moving
                    model_matrix = extrinsic2ModelView(rvec[0], tvec[0])
                else:
                    model_matrix = self.pre_extrinsicMatrix
            else:
                model_matrix =  self.pre_extrinsicMatrix
            
                
            if model_matrix is not None:     
                self.pre_extrinsicMatrix = model_matrix
                glLoadMatrixf(model_matrix)
                glScaled(self.model_scale, self.model_scale, self.model_scale)

                last_point = [0,0,0]
                scale = 1000/(self.model_scale/0.001)
                for point in points:
                    cur_point = [point[0][0]*scale,point[1][0]*scale,point[2][0]*scale]

                    glTranslatef(cur_point[0]-last_point[0],cur_point[1]-last_point[1],cur_point[2]-last_point[2])
                    #glRotatef(90, 1, 0, 0)
                    glCallList(self.model.gl_list)
                    last_point = cur_point

        image = cv2.resize(image, (self.out_width, self.out_height), interpolation=cv2.INTER_LINEAR)                
        cv2.imshow("Frame",image)
        cv2.waitKey(20)
        self.points_frames.append(image)
   
        
    def run(self):
        # Begin to render
        glutMainLoop()
  

if __name__ == "__main__":
    cv_file = cv2.FileStorage("./charuco_camera_calibration.yaml", cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("camera_matrix").mat()
    dist = cv_file.getNode("dist_coeff").mat()
    obj_path = './Models/Panels/SolarPanel.obj'
    building_path = './Building.txt'
    ar_instance = AR_render(mtx, dist, obj_path, building_path, model_scale = 0.005) #make it to mm
    ar_instance.run() 