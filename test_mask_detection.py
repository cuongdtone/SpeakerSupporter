# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import sys
import numpy as np
import datetime
import os
import glob
from utils.face_detecter_cov import RetinaFaceCoV

thresh = 0.7
mask_thresh = 0.8
scales = [640, 640]

count = 1

gpuid = -1
detector = RetinaFaceCoV('src/cov2_models/mnet_cov2', 0, gpuid, 'net3l')

cap = cv2.VideoCapture(0)
while True:
        _, img = cap.read()
        faces, landmarks = detector.detect(img,
                                           thresh
                                           )
        for face in faces:
            box = face[:4].astype('int')
            if face[5]>mask_thresh:
                color = [0, 0, 255]
            else:
                color = [0, 255, 0]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

        cv2.imshow('cov test', img)
        cv2.waitKey(2)
