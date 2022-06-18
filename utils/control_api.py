# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

from pynput.keyboard import Key, Controller
import time

# time.sleep(3)
keyboard = Controller()

# Press and release space aAAWorld
# keyboard.press(Key.right)
# keyboard.release(Key.right)

def process_event(event):
    if event == 'next':
        print(event)
        keyboard.press(Key.right)
        keyboard.release(Key.right)
    elif event == 'back':
        print(event)
        keyboard.press(Key.left)
        keyboard.release(Key.left)
