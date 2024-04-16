import pygame
import serial
import struct
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

long  = 1
width = 0.8
hight = 0.2

# Cube vertices and faces
vertices = [
    [long, -width, -hight], [long, width, -hight], [-long, width, -hight], [-long, -width, -hight],
    [long, -width, hight], [long, width, hight], [-long, width, hight], [-long, -width, hight]
]

face_edges = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 4, 7, 3),
    (1, 5, 6, 2),
    (0, 1, 5, 4),
    (3, 2, 6, 7)
)

line_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [4, 5], [5, 6], [6, 7], [7, 4]
]

blue_gray   = (0.68, 0.84, 1)
line_white  = (1,1,1)

face_colors = [blue_gray for i in range(6)]

# kalman filter variables
from filterpy.kalman import KalmanFilter

# 初始化卡尔曼滤波器
kf = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)

# 初始化状态转移矩阵
kf.F = np.eye(3)

# 初始化测量矩阵
kf.H = np.eye(3)

# 初始化过程噪声协方差矩阵
kf.Q = np.eye(3)

# 初始化测量噪声协方差矩阵
kf.R = np.diag([0.1, 0.1, 0.1])

# 初始化状态估计矩阵
kf.x = np.array([[0], [0], [0]])

# 初始化状态估计协方差矩阵
kf.P = np.eye(3)

# 初始化控制输入矩阵
kf.B = np.eye(3)

# 初始化控制输入
# u = np.array([[x_gyro], [y_gyro], [z_gyro]])

def draw_cube():
    glBegin(GL_QUADS)
    for face in range(len(face_edges)):
        glColor3fv(face_colors[face])
        for vertex in face_edges[face]:
            glVertex3fv(vertices[vertex])
    glEnd()
    
    glColor3f(line_white[0], line_white[1], line_white[2])
    glBegin(GL_LINES)
    for edge in line_edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    # Connect to the serial port
    ser = serial.Serial('COM7', 115200)  # Replace 'COM3' with your serial port and baud rate

    rawFrame = []

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Read gyroscope and accelerometer measurements from the serial port

        byte  = ser.read(1)        
        rawFrame += byte 
        
        if rawFrame[-2:]==[13, 10]:
            if len(rawFrame) == 14:
                            
                (x_gyro, y_gyro, z_gyro, x_acc, y_acc, z_acc) = struct.unpack('>hhhhhh', bytes(rawFrame[:-2]))
                        
                # debug info
                output = 'gyr_x={0:<9} gyr_y={1:<9} gyr_z={2:<6} acc_x={3:<6} acc_y={4:<6} acc_z={5:<6}'.format(
                    x_gyro,
                    y_gyro,
                    z_gyro,
                    x_acc,
                    y_acc,
                    z_acc
                )
                
            rawFrame = []
            print(x_gyro,y_gyro,z_gyro,x_acc,y_acc,z_acc)
            # GYRO_RESOLUATION_0 = 16.4
            GYRO_RESOLUATION_1 = 32.8
            # GYRO_RESOLUATION_2 = 65.6
            # GYRO_RESOLUATION_3 = 131.2
            
            # GYRO_RESOLUATION_4 = 264.4
            gyro_reso = GYRO_RESOLUATION_1

            #acc_reso = 4096
            #x_acc = float(x_acc)/float(4096)
            #y_acc = float(y_acc)/float(4096)
            #z_acc = float(z_acc)/float(4096)

            DATA_INTERVAL = 0.0625

            x_gyro = DATA_INTERVAL*float(x_gyro)/float(gyro_reso)
            y_gyro = DATA_INTERVAL*float(y_gyro)/float(gyro_reso)
            z_gyro = DATA_INTERVAL*float(z_gyro)/float(gyro_reso)

            #roll = math.atan2(float(y_acc),float(z_acc))
            #pitch = -math.atan2(float(x_acc),((float(y_acc)**2 + float(z_acc)**2))**(0.5))
            # 预测下一时刻状态
            u = np.array([[x_gyro],[y_gyro],[z_gyro]]) 
            kf.predict(u=u)
            # 更新状态估计
            z = np.array([[x_acc],[y_acc],[z_acc]]) 
            kf.update(z=z)
            print('filtered:',kf.x[0],kf.x[1],kf.x[2])
            # 原始数据
            # glRotatef(x_gyro, 1, 0, 0)
            # glRotatef(y_gyro, 0, 1, 0)
            # glRotatef(z_gyro, 0, 0, 1)
            # 滤波数据
            glRotatef(kf.x[0], 1, 0, 0)
            glRotatef(kf.x[1], 0, 1, 0)
            glRotatef(kf.x[2], 0, 0, 1)
            #fliter part ends
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_cube()
            pygame.display.flip()
            pygame.time.wait(10)

    # Close the serial port when finished
    ser.close()

if __name__ == '__main__':
    main()
