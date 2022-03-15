import os
import numpy as np


import matplotlib.pyplot as plt
import open3d

from utils import *

view_points2d = cv2.projectPoints(view_points3d, rot_vec, t_vec, self.K, self.distortion)[0].reshape(-1, 2)
        p = view_points2d
        selection = np.all((p[:, 0] >= 0, p[:, 0] < img.shape[1], p[:, 1] >= 0, p[:, 1] < img.shape[0]), axis=0)
        p = p[selection]
        
def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img


if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000114_image.png')), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_calib_file('data/000114_calib.txt')

    # Load labels
    labels = load_label('data/000114_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan('data/000114.bin')[:, :3]

    # render_image_with_boxes(rgb, labels, calib)
    render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    # render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
