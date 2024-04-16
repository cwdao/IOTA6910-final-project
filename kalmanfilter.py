import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy.random import randn

from math import radians, sin, cos
import math


# # Kalman filter parameters
# initial_state = position[0]
# initial_estimate_error = 1
# process_variance = 0.1
# measurement_variance = 0.5
# measurements = sensor

dt = 0.2
# fx.vel = math.cos(theta) * velocity
# fy.vel = math.sin(theta) * velocity
# ball_pos_x_init = 0
# ball_pos_y_init = 1
# ball_pos_velosity_init = 100
# ball_pos_theta_deg = 60
pos_init_x = ball_pos_x_init
pos_init_y = ball_pos_y_init
theta = math.radians(ball_pos_theta_deg)
vel_init_x = math.cos(theta) * ball_pos_velosity_init
vel_init_y = math.sin(theta) * ball_pos_velosity_init
# init state
# 球的初始状态，[x位置，x速度,y位置，y速度]
X_init = np.array([[pos_init_x], [vel_init_x], [pos_init_y], [vel_init_y]])
# 过程协方差Q
Q_init = np.array([[0,0,0,0], [0,0,0,0],[0,0,0,0],[0,0,0,0]])
# 观测协方差R
R_init = np.array([[0.5,0], [0,0.5]])
# 初始估计误差/先验误差
P_init = np.array([[0.1, 0], [0, 0.1]]) 
# 状态转移矩阵
F_ = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]) 
# 控制矩阵
B_ = np.array([[0, 0],[0, 0],[0, 0],[0, dt]])
# 控制输入
u_ = np.array([[0],[-9.8]])
# 观测矩阵
H_ = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
#kalman filter
times = np.linspace(1,100,100)
position_real = np.linspace(0,99,100)
# 实际观测序列
# measurement_real = []
# for i in range(len(position_real)):
#     measurement_real.append( position_real[i] + np.random.normal(0,4))
# 实际观测初值
# init_position_guess = position_real[0] + np.random.normal(0,3)
# init_velosity_guess = 1.5 + np.random.normal(0,1)
# init_guess = np.array([[init_position_guess],[init_velosity_guess]])
# init_covariance = np.array([[3,0], [0,1]])

def kalman_filter(initial_state, initial_estimate_error, F,Q,R,H,B,u, measurements):
    num_measurements = len(measurements)
    state_estimate = initial_state
    covariance_estimate = initial_estimate_error
    filtered_states = []
    # filtered_states_position=[]
    filtered_states_position_y = []
    filtered_states_position_x = []
    filtered_covariances =[]

    for i in range(num_measurements):
        # Prediction step

        predict_state = F.dot(state_estimate) + B.dot(u)
        predict_covariance = F.dot(covariance_estimate).dot(F.T) + Q

        
        # Update step

        # y = measurements[i]- H.dot(predict_state)
        y = np.array([[measure_ball_x[i],measure_ball_y[i]]]).T- H.dot(predict_state)
        s = H.dot(predict_covariance).dot(H.T)+R
        kalman_gain = predict_covariance.dot(H.T).dot(np.linalg.inv(s))
        state_estimate = predict_state + kalman_gain.dot(y)
        covariance_estimate = (np.eye(4)-kalman_gain.dot(H)).dot(predict_covariance)
        
        filtered_states_position_x.append(state_estimate[0])
        filtered_states_position_y.append(state_estimate[2])
        filtered_states.append(state_estimate)
        # filtered_states_velosity.append(state_estimate[1,:])
        filtered_covariances.append(covariance_estimate)

    return filtered_states_position_x,filtered_states_position_y,filtered_covariances, filtered_states
    
# Kalman filter parameters
# initial_state = init_guess
initial_estimate_error = Q_init
# process_variance = 0.1
# measurement_variance = 0.5
measurements = np.array([[measure_ball_x,measure_ball_y]])

# Apply Kalman filter
filtered_states_x,filtered_states_y,filtered_cov, filtered_st = kalman_filter(initial_state=X_init, initial_estimate_error=initial_estimate_error, F=F_,Q=Q_init,R=R_init,H=H_, B=B_,u=u_,measurements= measurements)
# filtered_position = filtered_states
# Plot the results
plt.figure(figsize=(10, 6))
# plt.plot(position_real, label='True Location', linestyle='dashed')
plt.scatter(measure_ball_x,measure_ball_y,marker='.',label='measure Location')
plt.plot(filtered_states_x,filtered_states_y, label='Prediction', linestyle='dotted')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('2dKalman Filter')
plt.show()

