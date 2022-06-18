# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import time

class Action():
    def __init__(self, hands_estimater, speaker=[]):
        self.hands = hands_estimater
        self.speaker = None
        if self.speaker is None and len(speaker)>0:
            self.speaker = speaker[0]
        self.last_action = None
        self.last_time = time.time()
        self.delay_action = 3

    def workflow(self, frame, speaker): # speaker is list of name
        speaker = [i['fullname'] for i in speaker if i['position']=='speaker']
        if len(speaker) > 0:
            self.speaker = speaker[0] # add speaker
            # hello speaker
        elif len(speaker)==0:
            return frame, None, None
        frame, action, conf = self.hands.predict(frame)
        if action is None:
            # print('1')
            return frame, None, None
        if self.last_action == action and time.time() - self.last_time < self.delay_action:
            # print('11')
            return frame, None, None
        self.last_action = action
        self.last_time = time.time()
        return frame, self.speaker, action



