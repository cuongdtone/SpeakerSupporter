# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import cv2
import yaml

config = yaml.load(open('src/setting.yaml'), yaml.FullLoader)
camera = 'id_camera_2'
cap = cv2.VideoCapture(config[camera])
ret, frame = cap.read()

cv2.namedWindow('cc', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cc', 1280, 720)
r = cv2.selectROI('cc', frame)

y1 = int(r[1])
y2 = int(r[1]+r[3])
x1 = int(r[0])
x2 = int(r[0]+r[2])

roi = [x1, y1, x2, y2]
config.update({'roi_1': roi} if camera=='id_camera_1' else {'roi_2': roi})

with open('src/setting.yaml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)