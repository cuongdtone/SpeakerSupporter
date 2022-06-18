# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import mediapipe as mp
import numpy as np
# from hands_utils import CentroidTracker
from tensorflow.keras.models import load_model

class HandsMotions():
    def __init__(self, weights='src/model.h5'):
        self.actions = ['next', 'back', 'nothing']
        self.seq_length = 10
        self.model = load_model(weights)
        self.threshold = 0.8
        # self.track = CentroidTracker()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.seq1 = []
        self.actions_seq1 = []
        self.seq2 = []
        self.actions_seq2 = []

    def predict(self, frame):
        result = self.hands.process(frame[..., ::-1])
        if result.multi_hand_landmarks is not None:

            for idx, res in enumerate(result.multi_hand_landmarks):
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])
                # centroid = (int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20))

                self.seq1.append(d)
                self.mp_drawing.draw_landmarks(frame, res, self.mp_hands.HAND_CONNECTIONS)

        if len(self.seq1) < self.seq_length:
            return frame, None, None
        input_data = np.expand_dims(np.array(self.seq1[-self.seq_length:], dtype=np.float32), axis=0)
        y_pred = self.model.predict(input_data).squeeze()
        self.seq1 = self.seq1[-self.seq_length+3:]

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]
        action = self.actions[i_pred]
        if conf > self.threshold:
            self.actions_seq1.append(action)
        if len(self.actions_seq1) < 3:
            return frame, None, None
        this_action = '?'
        if self.actions_seq1[-1] == self.actions_seq1[-2] == self.actions_seq1[-3]:
            this_action = action
        return frame, this_action, conf

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hands_motion = HandsMotions('/home/cuong/Desktop/presenter_support/src/hand_motion.h5')
    while True:
        ret, frame = cap.read()
        frame, action, confiden = hands_motion.predict(frame)
        if action is not None and action is not '?':
            print(action, confiden)

        cv2.imshow('cc', frame)
        cv2.waitKey(2)

