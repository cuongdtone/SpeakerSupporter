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

gpuid = -1
# detector = RetinaFaceCoV('src/cov2_models/mnet_cov2', 0, gpuid, 'net3l')
detector = RetinaFace('src/det_500m.onnx')

list_img = glob.glob('testset_facemask/obj_train_data/*.[jp][pn]*')
print(len(list_img))


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    boxes_xyxy = []

    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])
    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)
    # 2. calculate the area of inters
    inters = iw*ih
    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)
    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou


truth_vector = []
pred_vector = []
c = 0
for i in list_img:
    image = cv2.imread(i)

    faces, kpss = detector.detect(image)
    for face in faces:
        box = face[:4].astype('int')
        color = [0, 255, 0]
        # if face[5] > 0.5:
        #     color = [0, 0, 255]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

    # cv2.imshow('cov test', image)
    # cv2.waitKey(2)
    # print(faces)

    #
    h, w = image.shape[:2]
    path_label = '.'.join(i.split('.')[-2:-1]) + '.txt'
    truth = np.loadtxt(path_label, delimiter=' ')
    try:
        box_truth = xywh2xyxy(truth[:, 1:])
        label = truth[:, 0]
    except:
        box_truth = xywh2xyxy(np.array([truth[1:]]))
        label = [truth[0]]
    box_truth[:, [0, 2]] = box_truth[:, [0, 2]] * w
    box_truth[:, [1, 3]] = box_truth[:, [1, 3]] * h

    # print(box_truth)
    # print('-'*20)
    if len(faces)>len(box_truth):
        for _ in range(-len(box_truth) + len(faces)):
            truth_vector.append(1)
            pred_vector.append(0)

    for i, box in enumerate(box_truth):
        box_t = box.astype(np.int)
        label_t = 0

        truth_vector.append(int(label_t))

        have = 0
        for face in faces:
            box_p = face[:4].astype('int')
            label_p = 0 #face[5] + 1
            # is bg
            iou = get_iou(box_t, box_p)
            if iou>0.3:
                pred_vector.append(label_p)
                have = 1
                # print(iou)
                break

        if label[i]==0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        box = box.astype(np.int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        if have == 0:
            c+=1
            pred_vector.append(1)
            # cv2.imshow('cc', image)
            # cv2.waitKey()
    #
# print(truth_vector)
# print(pred_vector)

acc = metrics.accuracy_score(pred_vector, truth_vector)
print(acc)
print(c)

cm = metrics.confusion_matrix(pred_vector, truth_vector)
plot_cm(cm, normalize=False, title='Accuracy: %.2f%%'%(acc*100))