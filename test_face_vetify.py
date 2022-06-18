# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import glob
import shutil
import os
import numpy as np
import cv2
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX
from utils.face_aligner import get_similarity_transform
from utils.sqlite_db import insert_casia, get_casia, str2list

detector = RetinaFace('src/det_500m.onnx')
recognizer = ArcFaceONNX('src/w600k_mbf.onnx')

#phase 1: create test set structure from casia ( update not use)
# list_person = glob.glob('test_data/casia_face/CASIA-WebFace/*') #.[jp][pn]*')
# save_path = 'test_data/casia_face/test_set/'
# print(len(list_person))
#
# for i in list_person:
#     img_001_path = i + '/001.jpg'
#     des = save_path + i.split('/')[-1] + '.jpg'
#     shutil.move(img_001_path, des)


"""#phase 2: create and compress face embedding"""
# list_person = glob.glob('test_data/casia_face/CASIA-WebFace/*')
# list_person.sort()
#
# mat_feat = []
# for person in list_person:
#     list_img = glob.glob(person + '/*.[jp][pn]*')
#     list_img.sort()
#
#     feats = [] #one person
#     for c, img in enumerate(list_img):
#         if c>10:
#             break
#         image = cv2.imread(img)
#         faces, kpss = detector.detect(image)
#         try:
#             kps = kpss[0]
#             feat = recognizer.face_encoding(image, kps)
#             # print(feat.shape)
#             feats.append(feat)
#             # cv2.imshow('cc', warped)
#             # cv2.waitKey()
#         except:
#             pass
#     if len(feats)>0:
#         feet = np.sum(np.array(feats), axis=0) / len(feats)
#         id = person.split('/')[-1]
#         insert_casia(id, str(feet))
#     else:
#         print('None')



"""Phase 3: load test db and read all test acess to draw accuracy. Method 1: with Loop search method"""
"""# Do chinh xac"""
# raw_data = get_casia(None)
# data = np.zeros((len(raw_data), 513))
# print(data.shape)
# for idx, d in enumerate(raw_data):
#     data[idx, 0] = d['id']
#     feat = np.array(str2list(d['embed']))
#     data[idx, 1:] = feat
#
# # print(data[0])
# list_person = glob.glob('test_data/casia_face/CASIA-WebFace/*')
# list_person.sort()
#
# y_truth = []
# y_pred = []
# for person in list_person:
#     id_truth = float(person.split('/')[-1])
#
#     list_img = glob.glob(person + '/*.[jp][pn]*')
#     list_img.sort()
#     # print('-'*10)
#     for idx, img in enumerate(list_img[::-1]):
#         if idx>5: break
#         image = cv2.imread(img)
#         faces, kpss = detector.detect(image)
#         try:
#             kps = kpss[0]
#             feat = recognizer.face_encoding(image, kps)
#
#             max_sim = 0
#             id_pred = 0.0
#             for one_data in data:
#                 id_db = one_data[0]
#                 feat_db = one_data[1:]
#                 sim = recognizer.compute_sim(feat, feat_db)
#                 if sim>0.5 and sim>max_sim:
#                     id_pred = id_db
#             if id_pred != 0.0:
#                 y_truth.append(id_truth)
#                 y_pred.append(id_pred)
#             # if id_pred == id_truth:
#             #     print('True')
#             #
#             # else:
#             #     print('False')
#             # if id_pred == 0.0:
#             #     print('uknown')
#         except:
#             # print('Pass')
#             pass
# from sklearn.metrics import accuracy_score
# print(y_truth)
# print(y_pred)
# print(accuracy_score(y_truth, y_pred))

"""#Do nhay"""
raw_data = get_casia(None)
data = np.zeros((len(raw_data), 513))
print(data.shape)
for idx, d in enumerate(raw_data):
    data[idx, 0] = d['id']
    feat = np.array(str2list(d['embed']))
    data[idx, 1:] = feat

# print(data[0])
list_person = glob.glob('test_data/casia_face/CASIA-WebFace/*')
list_person.sort()

y_truth = []
y_pred = []
for d, person in enumerate(list_person):
    if d == 10: break
    id_truth = float(person.split('/')[-1])

    list_img = glob.glob(person + '/*.[jp][pn]*')
    list_img.sort()
    # print('-'*10)
    for idx, img in enumerate(list_img[::-1]):
        # if idx>5: break
        image = cv2.imread(img)
        faces, kpss = detector.detect(image)
        try:
            kps = kpss[0]
            feat = recognizer.face_encoding(image, kps)

            max_sim = 0
            id_pred = 0.0
            for one_data in data:
                id_db = one_data[0]
                feat_db = one_data[1:]
                sim = recognizer.compute_sim(feat, feat_db)
                if sim>0.5 and sim>max_sim:
                    id_pred = id_db
            # if id_pred != 0.0:
            y_truth.append(id_truth)
            y_pred.append(id_pred)
            # if id_pred == id_truth:
            #     print('True')
            #
            # else:
            #     print('False')
            # if id_pred == 0.0:
            #     print('uknown')
        except:
            # print('Pass')
            pass
from sklearn.metrics import accuracy_score, confusion_matrix
y_truth = np.array(y_truth)
y_pred = np.array(y_pred)
y_truth[y_truth>0] = 1
y_pred[y_pred>0] = 1
y_truth = y_truth.astype(np.bool)
y_pred = y_pred.astype(np.bool)
print(y_truth)
print(y_pred)
acc = accuracy_score(y_truth, y_pred)
print(accuracy_score(y_truth, y_pred))
cm = confusion_matrix(~y_pred, ~y_truth)
from utils.plot import plot_cm
plot_cm(cm, normalize=False, title='Accuracy %.2f%%'%(acc*100))

""" Test delay"""
import time


raw_data = get_casia(None)
data = np.zeros((len(raw_data), 513))
print(data.shape)
for idx, d in enumerate(raw_data):
    data[idx, 0] = d['id']
    feat = np.array(str2list(d['embed']))
    data[idx, 1:] = feat

# print(data[0])


image = cv2.imread('test_data/casia_face/test_set/0000108.jpg')

start_time = time.time()

faces, kpss = detector.detect(image)
kps = kpss[0]
feat = recognizer.face_encoding(image, kps)

time_compare = time.time()
max_sim = 0
id_pred = 0.0
for one_data in data:
    id_db = one_data[0]
    feat_db = one_data[1:]
    sim = recognizer.compute_sim(feat, feat_db)
    if sim>0.5 and sim>max_sim:
        id_pred = id_db
end_time = time.time()

print('Total time per frame: %.2f'%((end_time-start_time)*1000), ' ms')
print('FPS: %.2f'%(1/(end_time-start_time)), ' FPS')





