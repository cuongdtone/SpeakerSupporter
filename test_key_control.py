# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

from pynput.keyboard import Key, Controller
import time
time.sleep(3)
keyboard = Controller()

# Press and release space aAAWorld
keyboard.press(Key.right)
keyboard.release(Key.right)

# Type 'Hello World' using the shortcut type method
# keyboard.type('Hello World')

# from pynput import keyboard
#
# # The event listener will be running in this block
# while True:
#     with keyboard.Events() as events:
#         # Block at most one second
#         event = events.get(1.0)
#         if event is None:
#             print(event)
#         else:
#             print('Received event {}'.format(event))

# import cv2
#
# while True:
#     key = cv2.waitKey(4)
#     print(key)