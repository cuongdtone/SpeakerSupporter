# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import time
import cv2
import os
from utils.sqlite_db import insert_speaker, connect_database
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
import argparse
import glob
import numpy as np


id = input('ID : ')
fullname = input('fullname: ')
sex = int(input('Sex (0: male, 1: female) : '))
position = input('Position: ')

camera = 0

def save_new_image(dir, image):
    c = 1
    while os.path.exists(os.path.join(dir, '%d.jpg'%(c))):
        c += 1
    cv2.imwrite(os.path.join(dir, '%d.jpg'%(c)), image)


face_detect = RetinaFace(model_file='src/det_500m.onnx')
face_recognize = ArcFaceONNX(model_file='src/w600k_mbf.onnx')


def create_feat(db_path, id_code):
    path_to_dir = os.path.join(db_path, id_code)
    # position = input("Position:  ")
    # office = input("Office:  ")
    list_img = glob.glob(os.path.join(path_to_dir, '*.jpg')) + \
               glob.glob(os.path.join(path_to_dir, '*.jpeg')) + \
               glob.glob(os.path.join(path_to_dir, '*.png'))
    feets = []
    for i in list_img:
        image = cv2.imread(i)
        try:
            faces, kpss = face_detect.detect(image, max_num=0, metric='default', input_size=(640, 640))
            feet = face_recognize.face_encoding(image, kpss[0])
            feets.append(feet)
        except:
            continue
    feet = np.sum(np.array(feets), axis=0) / len(feets)
    return feet


db_root = 'src/data'
person = os.path.join(db_root, id)
try:
    os.mkdir(person)
except:
    pass

Cap = True
if Cap:
    cap = cv2.VideoCapture(camera)
    ret, frame = cap.read()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break
        cv2.imshow('ROI selector', frame)
        key = cv2.waitKey(5)
        if key == ord(' '):
            save_new_image(person, frame)
            time.sleep(0.3)
        elif key == ord('q'):
            break
    cap.release()


feat = create_feat(db_root, id)

print(feat)

# Phase 3: update DB
# update_face_feature_employee(None, id, str(feat))
db = connect_database()
insert_speaker(db, id, fullname, sex, position, str(feat))
# Phase 4: delete img dir