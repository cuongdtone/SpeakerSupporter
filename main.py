# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import numpy as np
import yaml
from utils.sqlite_db import load_user_data
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
from utils.human_landmarks import Human_Pose
from utils.face_landmark import TDDFA_ONNX
from utils.hands_motion import HandsMotions
from utils.control_api import process_event
from utils.sqlite_db import insert_timekeeping
from utils.face_threading import CameraSpeakerDetectThread, CameraParticipantsDetectThread
from utils.action_threading import Action


class Main():
    def __init__(self):
        # self.y5 = Y5Detect(weights='yolov5n.pt')
        self.load()
        # self.human_detecter = Y5Detect(weights='yolov5n.pt')
        self.face_detecter = RetinaFace(model_file='src/det_500m.onnx')
        # self.face_detecter2 = RetinaFaceCoV('src/cov2_models/mnet_cov2', 0, -1, 'net3l')
        self.face_recognizer = ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        self.human_pose = Human_Pose()
        self.tddfa = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))
        self.hands = HandsMotions(weights='src/hand_motion.h5')

        config = yaml.load(open('src/setting.yaml'), yaml.FullLoader)
        self.camera_speaker_detect = CameraSpeakerDetectThread(rtsp=config['id_camera_1'],
                                                               roi=config['roi_1'],
                                                               human_detecter=None,
                                                               face_detecter=self.face_detecter,
                                                               face_recognizer=self.face_recognizer,
                                                               face_landmark=self.tddfa,
                                                               speaker_data=self.speaker_data,
                                                               search_tree=self.search_tree)
        self.Action = Action(self.hands)
        self.camera_participants_detect = CameraParticipantsDetectThread(rtsp=config['id_camera_2'],
                                                                         roi=config['roi_2'],
                                                                         human_detecter=None,
                                                                         face_detecter=self.face_detecter,
                                                                         face_recognizer=self.face_recognizer,
                                                                         face_landmark=self.tddfa,
                                                                         speaker_data=None)

    def load(self):
        self.speaker_data, self.search_tree, self.speaker_info = load_user_data()

    def run(self):
        if self.camera_speaker_detect.check_cam():
            cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('cc', 1280, 720)
            data_queue, frame_queue = self.camera_speaker_detect.run()
        if self.camera_participants_detect.check_cam():
            cv2.namedWindow('c2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('c2', 1280, 720)
            data_2, frame_2 = self.camera_participants_detect.run()

        concentrate_track = []
        before_status = 'have speaker'
        while True:
            if self.camera_speaker_detect.check_cam():
                data = data_queue.get()
                frame = frame_queue.get()
                if self.camera_speaker_detect.track.have_speaker:
                    before_status = 'have'
                    try:
                        concentrate_track.append(count)
                    except:
                        pass
                else:
                    if before_status =='have':
                        print(self.camera_speaker_detect.track.timekeep)
                        print('count concentrate: ',concentrate_track)
                        if self.camera_speaker_detect.track.timekeep is not None:
                            code, fullname, position, from_time, to_time = self.camera_speaker_detect.track.timekeep
                            insert_timekeeping(code, fullname, position, from_time, to_time, np.array(concentrate_track))
                    concentrate_track = []
                    before_status = 'no'

                frame, speaker, action = self.Action.workflow(frame, data['people'])
                if action is not None:
                    process_event(action)
                cv2.imshow('cc', data['frame'])

            if self.camera_participants_detect.check_cam():
                data2 = data_2.get()
                count = data2['people']
                frame2 = frame_2.get()

                cv2.imshow('c2', data2['frame'])
            cv2.waitKey(2)


if __name__ == '__main__':
    Main().run()