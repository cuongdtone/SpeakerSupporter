# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
from utils.face_landmark import TDDFA_ONNX
from utils.face_pose import PoseEstimator

config = yaml.load(open('src/setting.yaml'), yaml.FullLoader)
detector = RetinaFace('src/det_500m.onnx')
recognizer = ArcFaceONNX('src/w600k_mbf.onnx')
landmarks_detector = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))
cap = cv2.VideoCapture(config['id_camera_2'])
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

        color = (0, 255, 0)
        concentrate = True
        rotation = rotation_vector.T[0]
        if (abs(rotation[0]) > 0.65):
            # Khong tap trung nhin sang 2 ben
            color = (0, 0, 255)
            concentrate = False
        if rotation[1] > 1.0:
            # Khong tap trung cui xuong
            color = (0, 0, 255)
            concentrate = False

        estimate_3d.append(rotation_vector.T[0].tolist())
        face_pose.draw_annotation_box(frame, rotation_vector, translation_vector, color=color)
        for kp in landmark.T:
            image = cv2.circle(frame, (int(kp[0]), int(kp[1])), 1, (0, 255, 0), 2)
    cv2.imshow('cc', frame)
    key = cv2.waitKey(2)
    if key == ord(' '):
        np.save('src/camera_view.npy', face_landmarks[0])
        print('saved')