#%%
import cv2
from cv2 import projectPoints
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import camproject



img = "/Users/Naren/Code/coding_challenge_perception/cameraimage.jpeg"
csv= "/Users/Naren/Code/coding_challenge_perception/pointcloud.csv"
saved_img= "/Users/Naren/Code/coding_challenge_perception/mapped_img.jpeg"
camera_calib= "/Users/Naren/Code/coding_challenge_perception/camera_calib.json"

columns= ["X", "Y", "Z", "intensity"]
df= pd.read_csv(csv,  dtype=np.float32, delimiter =',' ,header=None).T
df.columns= columns
scan= df.to_numpy()

points = scan[:, 0:3] # lidar xyz (front, left, up)velo

# velo = np.insert(points,3,1,axis=1).T

# velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
# velo =np.delete(velo, 3, axis=0)

#%%
file = open(camera_calib, 'r')
data = json.load(file)
rot_vec = cv2.Rodrigues(np.array([data['camera_roll_rad'],data['camera_pitch_rad'], data['camera_yaw_rad']]))
trans_vec= np.array([data['camera_x_m'], data['camera_y_m'], data['camera_z_m']])           

cam= camproject.Camera()
cam.intrinsics(1280, 960, 1613.33, 640, 480)
# R= cv2.Rodrigues(np.array([-0.012,0.02,0]))[0]
R_t= [[  0.9998000, -0.0001200,  0.0199982,1.95],
  [-0.0001200,  0.9999280,  0.0119989,0],
  [-0.0199982, -0.0119989,  0.9997280,1.29 ],
  [0,0,0,1]]

R_t2= [[  1, 0,  0,1.95],
  [0,  1,  0,0],
  [0, 0,  1,1.29 ],
  [0,0,0,1]]

R_t3=[[  0.9998000, -0.0001200,  0.0199982,0],
  [-0.0001200,  0.9999280,  0.0119989,0],
  [-0.0199982, -0.0119989,  0.9997280,0 ],
  [0,0,0,1]]

cam.attitudeMat(R_t3)
# cam.attitudeMat(R_t)
# K = np.array([[1613.33, 0 , 640],
#              [0, 1613.33, 480],
#              [0,0,1]])
#%%
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
#%%
points2= points.copy()


view_points2d = cam.project(points2) 
p = view_points2d
p= np.nan_to_num(p, copy=True, nan=0.0, posinf=None, neginf=None)

selection = np.all((p[:, 0] >= 0, p[:, 0] < png.shape[1], p[:, 1] >= 0, p[:, 1] < png.shape[0]), axis=0)
# selection = np.where((p[0, :] < IMG_W) & (p[0, :] >= 0) &
#                     (p[1, :] < IMG_H) & (p[1, :] >= 0) 
#                     # & (points[:, 2] > 0)
#                     )[0]

p = p[selection]
#%%
u,v = p.T
z= points2[:,2][selection]
plt.scatter(x=[u],y=[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.title("Mapping")
plt.savefig(saved_img,bbox_inches='tight')
plt.imshow(png)
plt.show()
  




# %%
