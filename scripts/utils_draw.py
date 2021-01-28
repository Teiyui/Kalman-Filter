#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np


# In[6]:


# 获取位置信息
def get_location(tmp):
    locations = []
    for i in range(len(tmp)):
        location = np.squeeze(tmp[i][0])
        locations.append(location)
    return locations


# In[7]:


# 获取速度信息
def get_speed(tmp):
    speeds = []
    for i in range(len(tmp)):
        speed = np.squeeze(tmp[i][1])
        speeds.append(speed)
    return speeds


# In[8]:


# 绘制位置比较图(卡尔曼滤波器)
def draw_location_kalman(actuals, measurements, priors, posteriors):
    actual_locations = get_location(actuals)
    measurement_locations = get_location(measurements)
    prior_locations = get_location(priors)
    posterior_locations = get_location(posteriors)
    # 设置横纵坐标
    x1 = np.arange(0, 31)
    x2 = np.arange(1, 31)
    ym1 = actual_locations
    ym2 = measurement_locations
    ym3 = prior_locations
    ym4 = posterior_locations
    lines = plt.plot(x1, ym1, x2, ym2, x2, ym3, x1, ym4)
    # 设置线的属性
    plt.setp(lines[0], linewidth=1.5)
    plt.setp(lines[1], linewidth=1.5)
    plt.setp(lines[2], linewidth=1.5)
    plt.setp(lines[3], linewidth=1.5)
    # 线的标签
    plt.legend(('actual', 'measurement', 'prior', 'posterior'), loc='upper right')
    plt.title("comparison on location")
    plt.xlabel("step")
    plt.ylabel("location")
    plt.show()


# In[9]:


# 绘制位置比较图(无卡尔曼滤波器)
def draw_location(actuals, measurements, states):
    actual_locations = get_location(actuals)
    measurement_locations = get_location(measurements)
    state_locations = get_location(states)
    # 设置横纵坐标
    x1 = np.arange(0, 31)
    x2 = np.arange(1, 31)
    ym1 = actual_locations
    ym2 = measurement_locations
    ym3 = state_locations
    lines = plt.plot(x1, ym1, x2, ym2, x2, ym3)
    # 设置线的属性
    plt.setp(lines[0], linewidth=1.5)
    plt.setp(lines[1], linewidth=1.5)
    plt.setp(lines[2], linewidth=1.5)
    # 线的标签
    plt.legend(('actual', 'measurement', 'states'), loc='upper right')
    plt.title("comparison on location without kalman filter")
    plt.xlabel("step")
    plt.ylabel("location")
    plt.show()


# In[11]:


# 绘制速度比较图(卡尔曼滤波器)
def draw_speed_kalman(actuals, measurements, priors, posteriors):
    actual_speeds = get_speed(actuals)
    measurement_speeds = get_speed(measurements)
    prior_speeds = get_speed(priors)
    posterior_speeds = get_speed(posteriors)
    # 设置横纵坐标
    x1 = np.arange(0, 31)
    x2 = np.arange(1, 31)
    ym1 = actual_speeds
    ym2 = measurement_speeds
    ym3 = prior_speeds
    ym4 = posterior_speeds
    lines = plt.plot(x1, ym1, x2, ym2, x2, ym3, x1, ym4)
    # 设置线的属性
    plt.setp(lines[0], linewidth=1.5)
    plt.setp(lines[1], linewidth=1.5)
    plt.setp(lines[2], linewidth=1.5)
    plt.setp(lines[3], linewidth=1.5)
    # 线的标签
    plt.legend(('actual', 'measurement', 'prior', 'posterior'), loc='upper right')
    plt.title("comparison on speed")
    plt.xlabel("step")
    plt.ylabel("speed")
    plt.show()


# In[ ]:


# 绘制速度比较图(无卡尔曼滤波器)
def draw_speed(actuals, measurements, states):
    actual_speeds = get_speed(actuals)
    measurement_speeds = get_speed(measurements)
    state_speeds = get_speed(states)
    # 设置横纵坐标
    x1 = np.arange(0, 31)
    x2 = np.arange(1, 31)
    ym1 = actual_speeds
    ym2 = measurement_speeds
    ym3 = state_speeds
    lines = plt.plot(x1, ym1, x2, ym2, x2, ym3)
    # 设置线的属性
    plt.setp(lines[0], linewidth=1.5)
    plt.setp(lines[1], linewidth=1.5)
    plt.setp(lines[2], linewidth=1.5)
    # 线的标签
    plt.legend(('actual', 'measurement', 'states'), loc='upper right')
    plt.title("comparison on speed without kalman filter")
    plt.xlabel("step")
    plt.ylabel("speed")
    plt.show()

