# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

from utils.face_threading import CameraSpeakerDetectThread
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
from utils.sqlite_db import load_user_data
import cv2

face_detecter = RetinaFace(model_file='src/det_500m.onnx')
face_recognizer = ArcFaceONNX(model_file='src/w600k_mbf.onnx')

user_data, _ = load_user_data()

camera_detect = CameraSpeakerDetectThread(rtsp=0, face_detecter=face_detecter, face_recognizer=face_recognizer, speaker_data=user_data)

data_queue, frame_queue = camera_detect.run()
while camera_detect.check_cam():
    data = data_queue.get()
    frames = frame_queue.get()
    cv2.imshow('cc', data['frame'])
    cv2.waitKey(2)