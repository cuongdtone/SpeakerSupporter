# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18


import threading
import cv2


class bufferless_camera(threading.Thread):
    def __init__(self, rtsp_url, name='camera-buffer-cleaner-thread'):
        self.camera = cv2.VideoCapture(rtsp_url)
        self.last_frame = None
        super(bufferless_camera, self).__init__(name=name)
        self.start()

    def run(self):
        while self.check_cam():
            ret, self.last_frame = self.camera.read()

    def get_frame(self):
        return self.last_frame

    def check_cam(self):
        return self.camera.isOpened()

    def get_resolution(self):
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (height, width)


def get_person(bboxes, labels, scores, threshold=0.6, labels_idx=False):
    new_bboxes, new_labels, new_scores = [], [], []
    for idx, label in enumerate(labels):
        if label =="person" and scores[idx]>threshold:
            new_bboxes.append(bboxes[idx])
            new_scores.append(scores[idx])
            if not labels_idx:
                new_labels.append(label)
            else:
                new_labels.append(0)
    return new_bboxes, new_labels, new_scores


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


