# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import mediapipe as mp
from utils.pose_utils import draw_landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class Human_Pose():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    def get(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        landmarks = draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return image[..., ::-1], landmarks


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    pose = Human_Pose()
    while True:
        ret, frame = cap.read()
        image, landmarks = pose.get(frame)

        print(landmarks.keys())
        cv2.imshow('cc', image)
        cv2.waitKey(2)