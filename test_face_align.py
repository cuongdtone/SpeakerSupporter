# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import sys
import numpy as np
import torch
import datetime
import os
import glob
from utils.face_detecter_cov import RetinaFaceCoV
from utils.face_detecter import RetinaFace
from sklearn import metrics
from utils.plot import plot_cm
from utils.face_aligner import get_similarity_transform, arcface_src
detector = RetinaFace('src/det_500m.onnx')

list_img = glob.glob('test_data/0000099/*.[jp][pn]*')
print(len(list_img))

cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cc', 480, 480)

c = 0
for i in list_img:
      image = cv2.imread(i)
      faces, kpss = detector.detect(image)
      try:
            kps = kpss[0]


            image_size = 112
            M = get_similarity_transform(kps)
            warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)

            ones = np.ones(shape=(len(kps), 1))
            points_ones = np.hstack([kps, ones])
            new_kps = M.dot(points_ones.T).T

            for kp in arcface_src[0]:
                  kp = kp.astype('int')
                  cv2.circle(warped, (kp[0], kp[1]), 2, (0, 255, 0), 1)

            for kp in new_kps:
                  kp = kp.astype('int')
                  cv2.circle(warped, (kp[0], kp[1]), 3, (0, 0, 255), 1)

            err = np.linalg.norm(arcface_src[0] - new_kps)
            print('Sai so bien doi: ', err)


            cv2.imshow('cc', warped)
            cv2.imshow('c2', image)
            key = cv2.waitKey()
            if key == ord('s'):
                  cv2.imwrite('test_data/save_test/' + str(c) + '_ori.jpg', image)
                  cv2.imwrite('test_data/save_test/' + str(c) + '_warped_%.2f.jpg'%(err), warped)
                  c+=1
      except Exception as e:
            pass



