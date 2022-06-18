# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import mediapipe as mp
from utils.pose_utils import draw_landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



class Hands():
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    def get(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        landmarks = []
        image.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark = draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                landmarks.append(landmark)
        return image[..., ::-1], landmarks


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    pose = Hands()
    while True:
        ret, frame = cap.read()
        image, landmarks = pose.get(frame)
        print(len(landmarks))

        cv2.imshow('cc', image)
        cv2.waitKey(2)