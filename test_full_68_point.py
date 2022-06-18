# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2.cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.face_detecter import RetinaFace
from utils.face_landmark import TDDFA_ONNX
from utils.face_pose import PoseEstimator
import yaml



def _get_full_model_points(filename='src/3d_face_points.txt'):
    """Get all 68 3D model points from file"""
    raw_value = []
    with open(filename) as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T

    # Transform the model into a front view.
    model_points[:, 2] *= -1

    return model_points

model_point = _get_full_model_points()/100


fig = plt.figure()
ax = plt.axes(projection='3d')

x = model_point[:, 0]
y = model_point[:, 1]
z = model_point[:, 2]

ax.plot3D(x, y, z, 'ro')
ax.set_title('Face 3d')



detector = RetinaFace('src/det_500m.onnx')
landmarks_detector = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))

image = cv2.imread('/home/cuong/Desktop/presenter_support/test_data/casia_face/test_set/0000156.jpg')
(h, w) = image.shape[:2]
face_pose = PoseEstimator(img_size=(h, w))
faces, kpss = detector.detect(image)
param_lst, roi_box_lst = landmarks_detector(image, faces)
face_landmarks = landmarks_detector.recon_vers(param_lst, roi_box_lst, dense_flag=False)
rotation_vector, translation_vector = face_pose.solve_pose_by_68_points(face_landmarks[0].T[:, :2])
# face_pose.draw_annotation_box(image, rotation_vector, translation_vector, color=[0, 0, 255])
landmark = face_landmarks[0].T
x = landmark[:, 0]
y = landmark[:, 1]
z = landmark[:, 2] + 50

z2 = np.zeros_like(z)

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot3D(x, y, z, 'ro')
ax2.plot3D(x, y, z2, 'go')
ax2.set_title('Face 3d')


for kp in landmark:
    image = cv2.circle(image, (int(kp[0]), int(kp[1])), 1, (0, 128, 0), 2)

cv2.cv2.namedWindow('cc', cv2.cv2.WINDOW_NORMAL)
cv2.cv2.resizeWindow('cc', 480, 480)
cv2.imshow('cc', cv2.cv2.flip(image, flipCode=1))
cv2.waitKey()
plt.show()