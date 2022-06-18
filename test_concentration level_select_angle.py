# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import numpy as np
import matplotlib.pyplot as plt

a = np.load('src/camera_view.npy')


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        self.model_points_68 = self._get_full_model_points()
        self.model_points_68_selective = np.load('src/camera_view.npy').T

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

        self.r_vec2 = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec2 = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename='src/3d_face_points.txt'):
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


    def solve_pose_selective(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        print(image_points.shape)
        print(self.model_points_68_selective.shape)
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68_selective,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec2,
            tvec=self.t_vec2,
            useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)



import shutil
import os
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
from utils.face_landmark import TDDFA_ONNX


detector = RetinaFace('src/det_500m.onnx')
landmarks_detector = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))


cap = cv2.VideoCapture(0) #'test_data/video1_clip1.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

face_pose = PoseEstimator(img_size=(height, width))

while cap.isOpened():
    _, frame = cap.read()
    faces, kpss = detector.detect(frame)
    param_lst, roi_box_lst = landmarks_detector(frame, faces)
    face_landmarks = landmarks_detector.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    for idx, face_box in enumerate(faces):
        cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (0,0, 255), 2)
        landmark = face_landmarks[idx]
        rotation_vector, translation_vector = face_pose.solve_pose_by_68_points(landmark.T[:, :2])

        rotation_vector2, translation_vector2 = face_pose.solve_pose_selective(landmark.T[:, :2])

        color = (0, 255, 0)
        concentrate = True
        rotation = rotation_vector2.T[0]
        if (abs(rotation[0]) > 0.65):
            # Khong tap trung nhin sang 2 ben
            color = (0, 0, 255)
            concentrate = False
        if rotation[1] > 1.0:
            # Khong tap trung cui xuong
            color = (0, 0, 255)
            concentrate = False

        face_pose.draw_annotation_box(frame, rotation_vector, translation_vector, color=color)
        for kp in landmark.T:
            image = cv2.circle(frame, (int(kp[0]), int(kp[1])), 1, (0, 255, 0), 2)
    cv2.imshow('cc', frame)
    key = cv2.waitKey(2)
    if key == ord(' '):
        print(face_landmarks)