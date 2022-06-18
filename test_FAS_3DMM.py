# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import glob
import shutil
import os
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
from utils.face_landmark import TDDFA_ONNX
from utils.face_pose import PoseEstimator


detector = RetinaFace('src/det_500m.onnx')
recognizer = ArcFaceONNX('src/w600k_mbf.onnx')
landmarks_detector = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))

cap = cv2.VideoCapture(0) #'test_data/video1_clip1.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

face_pose = PoseEstimator(img_size=(height, width))

estimate_3d = []
c = 0
while cap.isOpened():
    _, frame = cap.read()
    faces, kpss = detector.detect(frame)
    param_lst, roi_box_lst = landmarks_detector(frame, faces)
    face_landmarks = landmarks_detector.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    for idx, face_box in enumerate(faces):
        cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (0,0, 255), 2)
        landmark = face_landmarks[idx]
        rotation_vector, translation_vector = face_pose.solve_pose_by_68_points(landmark.T[:, :2])
        estimate_3d.append(rotation_vector.T[0].tolist())
        face_pose.draw_annotation_box(frame, rotation_vector, translation_vector, color=[0, 0, 255])
        for kp in landmark.T:
            image = cv2.circle(frame, (int(kp[0]), int(kp[1])), 1, (0, 255, 0), 2)
    if len(estimate_3d)>32:
        estimate_3d = np.array(estimate_3d)
        std_h = np.std(estimate_3d[:, 0])
        mean_h =  np.mean(estimate_3d[:, 0])
        std_v = np.std(estimate_3d[:, 1])
        mean_v = np.mean(estimate_3d[:, 1])

        plt.subplot(1, 2, 1)
        plt.plot(estimate_3d[:, 0])
        plt.title('Horizontal, mean = %.2f, std = %.2f'%(mean_h, std_h))
        plt.xlabel('frame')
        plt.ylabel('angle')
        plt.axis([0, 32, -2, 2])
        plt.subplot(1, 2, 2)
        plt.plot(estimate_3d[:, 1])
        plt.title('Vertical, mean = %.2f, std = %.2f'%(mean_v, std_v))
        plt.xlabel('frame')
        plt.ylabel('angle')
        plt.axis([0, 32, -2, 2])
        plt.pause(0)
        estimate_3d = []
    cv2.imwrite(f'FAS_frame/{c}.jpg', frame)
    c+=1
    cv2.imshow('cc', frame)
    cv2.waitKey(2)