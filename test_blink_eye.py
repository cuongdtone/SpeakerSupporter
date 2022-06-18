# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import numpy as np
from utils.face_detecter_cov import RetinaFaceCoV
from utils.face_detecter import RetinaFace
from utils.eye_classifier import Eye
import torch


eye = Eye(model_file='src/eye_blink/best.h5')
face_detecter2 = RetinaFace(model_file='src/det_500m.onnx') #RetinaFace('src/cov2_models/mnet_cov2', 0, -1, 'net3l')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces, kpss = face_detecter2.detect(frame)
    for idx, face in enumerate(faces):
        face = face.astype(np.int)
        pred, left, right = eye.predict(frame, kpss[idx])
        print(pred)
        cv2.imshow('c1', left)
        cv2.imshow('c2', right)
    cv2.imshow('cc', frame)
    cv2.waitKey(2)

