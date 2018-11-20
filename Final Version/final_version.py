
# coding: utf-8

# In[1]:

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import numpy as np
import time
import glob
import math
import graphics
import sys
import random
import os.path as osp
import pygame
import sys


# In[2]:

DEFAULT_THRESHOLD = 32
m_pre = None
n_pre = None
# Load Sound Effect
pygame.init()
s = pygame.mixer.Sound('Dong.WAV')


# In[3]:

# Set Rhythm
pygame.mixer.pre_init(44100,16,2,4096)
pygame.init()
pygame.mixer.music.load("P5.mp3")
pygame.mixer.music.set_volume(0.5)
score = 0
speed = 27
move_buffer = int(1)
next_is_ka = 0


# In[4]:

# Set AR
aruco = cv2.aruco
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)


# In[5]:

# Set Side Strike Parameters
mask_h = 100
mask_w = 100
area_lim2 = 650
hit_side_count = 0
p_side_time = time.time()
n_side_time = time.time()
ka = pygame.mixer.Sound('Ka.WAV')


# In[6]:

# Camera Calibration
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


# In[7]:

alpha = mtx[0][0]
beta = mtx[1][1]
cx = mtx[0][2]
cy = mtx[1][2]


# In[8]:

def draw_background():
    ## Enable / Disable
    glDisable(GL_DEPTH_TEST)    # Disable GL_DEPTH_TEST
    glDisable(GL_LIGHTING)      # Disable Light
    glDisable(GL_LIGHT0)        # Disable Light
    glEnable(GL_TEXTURE_2D)     # Enable texture map

    ## init
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear Buffer
    glColor3f(1.0, 1.0, 1.0)    # Set texture Color(RGB: 0.0 ~ 1.0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    ## draw background
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glPushMatrix()
    glBegin(GL_QUADS)
    glTexCoord2d(0.0, 1.0)
    glVertex3d(-1.0, -1.0,  0)
    glTexCoord2d(1.0, 1.0)
    glVertex3d( 1.0, -1.0,  0)
    glTexCoord2d(1.0, 0.0)
    glVertex3d( 1.0,  1.0,  0)
    glTexCoord2d(0.0, 0.0)
    glVertex3d(-1.0,  1.0,  0)
    glEnd()
    glPopMatrix()

    ## Enable / Disable
    glEnable(GL_DEPTH_TEST)     # Enable GL_DEPTH_TEST
    glEnable(GL_LIGHTING)       # Enable Light
    glEnable(GL_LIGHT0)         # Enable Light
    glDisable(GL_TEXTURE_2D)    # Disable texture map


# In[9]:

def createCircularMask(h, w, center=None, radius=None):
    # create the mask with a empty circle inside 
    if center is None:
        center = [int(w/2), int(h/2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


# In[10]:

def draw():

    ret, img = cap.read() #read camera image
    
    frame = cv2.flip(img, 1)
    
    global m_pre
    global n_pre
    global speed
    global move_buffer
    global score
    global n_side_time
    global p_side_time
    global hit_side_count
    global area_lim2
    global next_is_ka
    
    hit_flag = None
    side_flag = None
    
    diff_distance = 1000
    
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    
    mask_map = np.zeros((h, w), np.uint8)
    
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('raw',frame_diff)
    _,fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5))
    
    try:
        #print(1)
        frame_show = frame.copy()
        dilate_mask = cv2.dilate(fgmask,kernel,iterations = 6)
        potential_stick = cv2.bitwise_and(frame_show,frame_show,mask = fgmask)
        
        resColored = cv2.bitwise_and(frame_show,frame_show,mask = dilate_mask)
        
        blur = cv2.blur(resColored,(3,3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower_range = np.array([2,0,0])
        upper_range = np.array([16,255,255])
        
        Hand_mask = cv2.inRange(hsv,lower_range,upper_range)
        filtered = cv2.GaussianBlur(Hand_mask, (15,15), 1)
        ret,thresh = cv2.threshold(filtered, 127, 255, 0)
        DeleteHand_mask = cv2.dilate(thresh,kernel,iterations = 10)
        cv2.imshow('DeleteHand_mask',DeleteHand_mask)
        
        [x_coor,y_coor] = np.where(DeleteHand_mask==255)
        
        dilate_mask[x_coor,y_coor] = 0
        
        
        _,contours,hierarchy= cv2.findContours(dilate_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        # Threshold = 1200
        if np.max(areas) > 1200:
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            for c in cnt:
                cv2.drawContours(frame_show, [c], 0, (0,255,0), 3)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_show,(x,y),(x+w,y+h),(0,255,0),2)
            m = int(x+ 0.5*w)
            n = int(y+ 0.5*h)
            cv2.circle(frame_show,(m,n), 25, (255,0,0), 10)
            
                
            # Set Flag and Play Sound Effect
            if m_pre is None and n_pre is None:
                m_pre = m
                n_pre = n
                    
            else:
                #if (n_pre-n>30):
                #    cv2.putText(frame_show,"Go Up", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                if (n_pre-n<-80):
                    cv2.putText(frame_show,"Hit", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                    s.play()
                    hit_flag = 1
                    
                    #print(n)
                n_pre = n
        cv2.imshow('raw',frame_show)
        
        
        
        # Jinxin Function
        #cap_top = cv2.VideoCapture(0)
        
        ret, frame = cap_top.read()
        cv2.rectangle(frame,(170,170),(400,400),(0,255,0),0)            
        cv2.imshow('frame', frame)
        crop_image  = frame[170:400, 170:400]
        output = crop_image .copy()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        
        ret, frame = cap_top.read()
        crop_image  = frame[170:400, 170:400]
        frame_diff = cv2.absdiff(crop_image, output)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        
        #cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
         


        #par = np.int64(scaler*r)
        retval, thresh = cv2.threshold(gray_diff, 128, 255, cv2.THRESH_BINARY)
        #thresh_cut = cv2.resize(thresh[y-par:y+par,x-par:x+par],(mask_h,mask_w))
            
        cv2.imshow('frame4',thresh)
                
        #detect_stick_side(thresh_cut)
        _,cnt,hierarchy= cv2.findContours(thresh.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        area_list = []
        for i in range(len(cnt)):
            area_list.append(cv2.contourArea(cnt[i]))
        max_area = max(area_list)
                #print(max_area)
        if max_area> area_lim2:
            n_side_time = time.time()
            hit_side_count  = hit_side_count+1
            time_diff = n_side_time - p_side_time
            #print (time_diff)
            #print (self.hit_side_count)
            if time_diff> 0.3:
                if hit_side_count > 2:
                    ka.play()
                    side_flag = 1
        
                            
                
        
            
        """
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                par = np.int64(scaler*r)
                retval, thresh = cv2.threshold(gray_diff, 128, 255, cv2.THRESH_BINARY)
                thresh_cut = cv2.resize(thresh[y-par:y+par,x-par:x+par],(mask_h,mask_w))
                #thresh_cut_inv = cv2.bitwise_not(thresh_cut)
                #thresh_original = cv2.resize(thresh[y-r:y+r,x-r:x+r],(mask_h,mask_w))
                cv2.imshow('frame4',thresh_cut)
                
                #detect_stick_side(thresh_cut)
                _,cnt,hierarchy= cv2.findContours(thresh_cut.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                area_list = []
                for i in range(len(cnt)):
                    area_list.append(cv2.contourArea(cnt[i]))
                max_area = max(area_list)
                #print(max_area)
                if max_area> area_lim2:
                    n_side_time = time.time()
                    hit_side_count  = hit_side_count+1
                    time_diff = n_side_time - p_side_time
                    #print (time_diff)
                    #print (self.hit_side_count)
                    if time_diff> 0.3:
                        if hit_side_count > 2:
                            ka.play()
                            side_flag = 1
        
                            
                
                    
        cv2.imshow('frame', frame)
        """
        
        
        
        
        
        
    except:
        pass
    
    
    # Aruco
    corners, ids, _ = aruco.detectMarkers(img, dictionary)
    rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, 8.0, mtx, dist)
    if not ids is None:
        aruco.drawAxis(img, mtx, dist, rvec[0], tvec[0], 8.0) # Draw Axis
        aruco.drawDetectedMarkers(img, corners) # Draw Square and IDs
    
       
    # Load Game Component before 3d model added intp screen
        
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #BGR-->RGB
    img = cv2.flip(img, 1)
    
    h, w = img.shape[:2]
    cv2.circle(img,(int(h/2),100), 25, (130,136,145), 10)
    cv2.putText(img,str(score), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    
    
    if hit_flag==1 and next_is_ka==0:
        
        cv2.circle(img,(int(h/2),100), 25, (255,0,0), 10)
        hit_position = int(w - int(speed*move_buffer))
        diff_distance = np.abs(hit_position - int(h/2))
        score = score + 2
        
    if side_flag==1 and next_is_ka==1:
        
        cv2.circle(img,(int(h/2),100), 25, (255,0,0), 10)
        hit_position_side = int(w - int(speed*move_buffer))
        diff_distance = np.abs(hit_position_side - int(h/2))
        score = score + 2
    
    if diff_distance > 70:
        if (int(w - int(speed*move_buffer)>0)):
            if next_is_ka == 0:
                cv2.circle(img,(int(w - int(speed*move_buffer)),100), 25, (0,0,255), 10)
            else:
                cv2.circle(img,(int(w - int(speed*move_buffer)),100), 25, (0,255,0), 10)
            move_buffer = move_buffer + 1
        else:
            cv2.putText(img,"Miss!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            move_buffer = 0
            speed = random.randint(10,25)
            score = score - 1
            next_is_what = random.randint(1,100)
            #print(next_is_what)
            if next_is_what > 50:
                next_is_ka = 1
            else:
                next_is_ka = 0
                
            #print(next_is_ka)
            
    else:
        move_buffer = 0
        speed = random.randint(10,25)
        cv2.putText(img,"Hit", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        next_is_what = random.randint(1,100)
        #print(next_is_what)
        if next_is_what > 60:
            next_is_ka = 1
        else:
            next_is_ka = 0
        #print(next_is_ka)
    
        
        
        
    # 3D painting
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    
    draw_background()

    ## make projection matrix
    f = 1000.0  #far
    n = 1.0     #near

    m1 = np.array([
    [(alpha)/cx, 0, 0, 0],
    [0, beta/cy, 0, 0],
    [0, 0, -(f+n)/(f-n), (-2.0*f*n)/(f-n)],
    [0,0,-1,0],
    ])
    glLoadMatrixd(m1.T)

    ## draw cube
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPushMatrix()
    
    # Change the color of Cube
    if hit_flag == 1:
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.9,0.1,0.1,1.0])
        #time.sleep(0.1) 
    else:
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8,0.7,0.6,1.0])
        
    # If find Tag    
    if not ids is None:
        # fix axis
        tvec[0][0][0] = tvec[0][0][0]
        tvec[0][0][1] = -tvec[0][0][1]
        tvec[0][0][2] = -tvec[0][0][2]

        rvec[0][0][1] = -rvec[0][0][1]
        rvec[0][0][2] = -rvec[0][0][2]
        m = compositeArray(cv2.Rodrigues(rvec)[0], tvec[0][0])
        glPushMatrix()
        glLoadMatrixd(m.T)

        glTranslatef(0, 0, -0.5)
        
        
        # Here Draw Model On the Tag
        
        #glutSolidCube(20.0)
        glRotatef(90, 1, 0, 0);
        glutSolidCone(20,30,16,16)
        #obj.render_scene()
        #obj.render_texture(surface_id,((0,0),(2,0),(2,2),(0,2)))
        glPopMatrix()

    glPopMatrix()

    # flush drawing routines to the window
    glFlush();
    glutSwapBuffers()


# In[ ]:

def compositeArray(rvec, tvec):
    v = np.c_[rvec, tvec.T]
    v_ = np.r_[v, np.array([[0,0,0,1]])]
    return v_

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    # Play Background Music
    pygame.mixer.music.play(-1)

def idle():
    glutPostRedisplay()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glLoadIdentity()
    glOrtho(-w / windowWidth, w / windowWidth, -h / windowHeight, h / windowHeight, -1.0, 1.0)

def keyboard(key, x, y):
    # convert byte to str
    key = key.decode('utf-8')
    if key == 'q':
        print('exit')
        sys.exit()


# In[ ]:

if __name__ == "__main__":
    # Set Camera
    windowWidth = 640
    windowHeight = 480
    cap = cv2.VideoCapture(0)
    cap_top = cv2.VideoCapture(1)
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap_top.set(cv2.CAP_PROP_FPS, 30)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, windowWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, windowHeight)
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInit(sys.argv)

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow(b"Drum")
    
    
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    init()
    glutIdleFunc(idle)

    glutMainLoop()


# In[ ]:



