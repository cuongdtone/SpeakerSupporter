# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue
from .functions import bufferless_camera, get_person, compute_color_for_labels
from .face_tracker import CentroidTracker, find_faces, TrackSpeaker
from .face_pose import PoseEstimator
from .sqlite_db import insert_timekeeping
import traceback


class CameraSpeakerDetectThread():
    def __init__(self, rtsp, roi, human_detecter=None, face_detecter=None, face_recognizer=None, face_landmark=None, speaker_data=None, search_tree=None):
        self.employees_data = speaker_data
        self.search_tree = search_tree
        self.human_detecter = human_detecter
        self.face_detecter = face_detecter #RetinaFace(model_file='src/det_500m.onnx')
        self.face_recognizer = face_recognizer #ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        self.face_landmark = face_landmark
        self.roi = roi
        self.cam_cleaner = bufferless_camera(rtsp)
        self.centroid_tracker = CentroidTracker()
        self.track = TrackSpeaker()
        self.face_pose = PoseEstimator(img_size=self.cam_cleaner.get_resolution())
        self.period = 3
        self.count = self.period

        self.frame_ori_queue = Queue(maxsize=3)
        self.data_detect_queue = Queue(maxsize=3)
        self.data_final_queue = Queue(maxsize=3)
        self.frame_final_queue = Queue(maxsize=3)

        self.detect = Thread(target=self.detect_thread, args=[self.frame_ori_queue, self.data_detect_queue])
        self.recognize = Thread(target=self.recognize_thread, args=[self.data_detect_queue, self.data_final_queue])
        self.detect.setDaemon(True)
        self.recognize.setDaemon(True)

    def check_cam(self):
        return self.cam_cleaner.camera.isOpened()

    def run(self):
        self.detect.start()
        self.recognize.start()
        return self.data_final_queue, self.frame_ori_queue

    def detect_thread(self, frame_ori_queue, data_detect_queue):
        while self.cam_cleaner.camera.isOpened():
            frame = self.cam_cleaner.last_frame
            if frame is not None:
                faces, kpss = self.face_detecter.detect(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :])

                put_data = {'frame': frame, 'faces': faces, 'kpss': kpss}
                data_detect_queue.put(put_data)
                frame_ori_queue.put(frame)

    def recognize_thread(self, data_detect_queue, data_final_queue):
        while self.cam_cleaner.camera.isOpened():
            data = data_detect_queue.get()
            frame = data['frame']
            faces = data['faces']
            kpss = data['kpss']

            objects, input_centroid = self.centroid_tracker.update(faces)
            out_info = []

            param_lst, roi_box_lst = self.face_landmark(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :], faces)
            face_landmarks = self.face_landmark.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            try:
                self.track.update()
                for idx, (objectID, centroid) in enumerate(objects.items()):
                    # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    face_box, kps, face_landmark = find_faces(objectID, objects, input_centroid, faces, kpss, face_landmarks)

                    color = compute_color_for_labels(objectID)
                    # if face_box[5]>0.6:
                    #     color = [0, 0, 255]
                    cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), color, 2)
                    # if face_box is None or face_box[5]>0.6:
                    #     continue

                    face_box[0] += self.roi[0]
                    face_box[1] += self.roi[1]
                    face_box[2] += self.roi[0]
                    face_box[3] += self.roi[1]
                    face_landmark.T[0] += self.roi[0]
                    face_landmark.T[1] += self.roi[1]
                    rotation_vector, translation_vector = self.face_pose.solve_pose_by_68_points(face_landmark.T[:, :2])

                    face_box = face_box.astype(np.int)
                    info = self.track.update(objectID, face_box, kps, frame, self.face_recognizer,
                                             self.employees_data, self.search_tree,
                                             self.count >= self.period, rotation_vector)
                    text = "{}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    if info is None:
                        text = 'Verifing'
                    else:
                        text = info['fullname'].split()[-1] + ' - %.2f' % (info['Sim'])
                        out_info.append(info)
                    t_size = cv2.getTextSize(text,
                                             fontFace=cv2.FONT_HERSHEY_PLAIN,
                                             fontScale=1.0, thickness=1)[0]
                    cv2.putText(frame, text, (int(face_box[0]), int(face_box[1] + t_size[1] + 5)), cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1.0, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    self.face_pose.draw_annotation_box(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :],
                                                       rotation_vector,
                                                       translation_vector,
                                                       color=color)
            except Exception as e:
                print(e)
                print(traceback.print_exc())
                pass
            if self.count >= self.period:
                self.count = 0
            else:
                self.count += 1
            cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (0, 255, 0), 2)
            data_final_queue.put({'frame': frame, 'people': out_info})


class CameraParticipantsDetectThread():
    def __init__(self, rtsp, roi, human_detecter=None, face_detecter=None, face_recognizer=None, face_landmark=None, speaker_data=None):
        self.employees_data = speaker_data
        self.human_detecter = human_detecter
        self.face_detecter = face_detecter #RetinaFace(model_file='src/det_500m.onnx')
        self.face_recognizer = face_recognizer #ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        self.face_landmark = face_landmark
        self.roi = roi
        self.cam_cleaner = bufferless_camera(rtsp)
        self.centroid_tracker = CentroidTracker()
        self.face_pose = PoseEstimator(img_size=self.cam_cleaner.get_resolution())

        self.frame_ori_queue = Queue(maxsize=2)
        self.data_detect_queue = Queue(maxsize=2)
        self.data_final_queue = Queue(maxsize=2)
        self.frame_final_queue = Queue(maxsize=2)

        self.detect = Thread(target=self.detect_thread, args=[self.frame_ori_queue, self.data_detect_queue])
        self.recognize = Thread(target=self.recognize_thread, args=[self.data_detect_queue, self.data_final_queue])
        self.detect.setDaemon(True)
        self.recognize.setDaemon(True)
        self.status = False

        self.concentrate_analysis = None

    def check_cam(self):
        return self.cam_cleaner.camera.isOpened()

    def run(self):
        self.detect.start()
        self.recognize.start()
        return self.data_final_queue, self.frame_ori_queue

    def detect_thread(self, frame_ori_queue, data_detect_queue):
        while self.cam_cleaner.camera.isOpened():
            frame = self.cam_cleaner.last_frame
            if frame is not None:
                faces, kpss = self.face_detecter.detect(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :])
                put_data = {'frame': frame, 'faces': faces, 'kpss': kpss}
                data_detect_queue.put(put_data)
                frame_ori_queue.put(frame)

    def recognize_thread(self, data_detect_queue, data_final_queue):
        while self.cam_cleaner.camera.isOpened():
            data = data_detect_queue.get()
            frame = data['frame']
            faces = data['faces']
            kpss = data['kpss']


            param_lst, roi_box_lst = self.face_landmark(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :], faces)
            face_landmarks = self.face_landmark.recon_vers(param_lst, roi_box_lst, dense_flag=False)

            number_person_concentrate = 0
            number_person_no_concentrate = 0
            try:
                objects, input_centroid = self.centroid_tracker.update(faces)
                for idx, (objectID, centroid) in enumerate(objects.items()):
                    # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    face_box, kps, face_landmark = find_faces(objectID, objects, input_centroid, faces, kpss, face_landmarks)
                    face_box[0] += self.roi[0]
                    face_box[1] += self.roi[1]
                    face_box[2] += self.roi[0]
                    face_box[3] += self.roi[1]
                    face_landmark.T[0] += self.roi[0]
                    face_landmark.T[1] += self.roi[1]
                    rotation_vector, translation_vector = self.face_pose.solve_pose_by_68_points(face_landmark.T[:, :2])
                    rotation_vector2, translation_vector2 = self.face_pose.solve_pose_by_68_points(face_landmark.T[:, :2], frontal_camera=False)
                    rotation = rotation_vector2.T[0]

                    concentrate = True
                    color = (0, 255, 0)
                    if (abs(rotation[0]) > 0.65):
                        # Khong tap trung nhin sang 2 ben
                        color = (0, 0, 255)
                        concentrate = False
                    if rotation[1]>1.0:
                        # Khong tap trung cui xuong
                        color = (0, 0, 255)
                        concentrate = False

                    if concentrate:
                        number_person_concentrate += 1
                    else:
                        number_person_no_concentrate += 1
                    text = "{}".format(objectID) + ' -h %.2f -v %.2f'%(rotation[0], rotation[1])
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    if face_box is None:
                        continue
                    # color = compute_color_for_labels(objectID)
                    self.face_pose.draw_annotation_box(frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2], :],
                                                       rotation_vector,
                                                       translation_vector,
                                                       color=color)
                    #cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 2)

                cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[2], self.roi[3]), (0, 255, 0), 2)
            except Exception as e:
                print(e)
            data_final_queue.put({'frame': frame, 'people': [number_person_concentrate, number_person_no_concentrate]})





