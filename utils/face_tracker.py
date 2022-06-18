# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18


import time
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from .face_recognizer import face_compare_tree
from datetime import datetime
import cv2


class CentroidTracker():
	def __init__(self, maxDisappeared=0):

		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in self.disappeared.keys():
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects, []

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY, scrore)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()


			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				# val
				if row in usedRows or col in usedCols:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		# faces = sort_box(self.objects, inputCentroids, rects)
		return self.objects, inputCentroids


def find_faces(id, objects, input_centroid, faces, kpss, landmarks):
	for idx, c in enumerate(input_centroid):
		if objects[id][0] == c[0] and objects[id][1] == c[1]:
			return faces[idx], kpss[idx], landmarks[idx]
	return None, None, None

def update_dict(dict: dict, pop_keys: list):
	new_dict = {}
	for i in dict.keys():
		if i not in pop_keys:
			new_dict.update({i: dict[i]})
	return new_dict

class TrackSpeaker():
	def __init__(self, img_size=(640, 480), time_track_fas=2, fps_process=10):
		self.people_tracked = {}
		self.frame_FAS = fps_process * time_track_fas
		self.frame_verify = 1
		self.verify_threshold = 0.7
		self.recog_frame = self.frame_verify

		self.have_speaker = False
		self.timekeep = None

		# self.eye = Eye()

	def update(self, id=None, box=None, kps=None, frame=None,
			   face_recognizer=None, employees_data=None, search_tree=None, recog=None, rotation_face=None):
		pop_list = []
		for i in self.people_tracked.keys():
			if datetime.now().second - self.people_tracked[i]['last_seen'].second >1: # no see 1s
				info = self.people_tracked[i]['verify']
				if info is not None:
					self.timekeep = (info['id'],
									 info['fullname'],
									 info['position'],
									 self.people_tracked[i]['first_seen'].strftime("%Y-%m-%d %H:%M:%S"),
									 self.people_tracked[i]['last_seen'].strftime("%Y-%m-%d %H:%M:%S"))
				pop_list.append(i)
				self.have_speaker = False
		self.people_tracked = update_dict(self.people_tracked, pop_list)
		if id is not None:
			if not id in self.people_tracked.keys(): # New
				self.people_tracked.update({id: {'box': box,
								  'kps': kps,
								  'FAS': 0,
								  'verify': None,
								  'FAS_track_frame':[],
								  'first_seen': datetime.now(),
								  'last_seen': datetime.now()}})
			else:
				# if box is None and kps is None:
				# 	self.people_tracked.pop(id)
				# 	return None
				# else: # Update old person
					# self.people_tracked[id] = [box, kps]
					# FAS method
				self.people_tracked[id]['last_seen'] = datetime.now()
				if self.people_tracked[id]['FAS'] == 0:
					if len(self.people_tracked[id]['FAS_track_frame']) >= self.frame_FAS:
						total_angle = np.array(self.people_tracked[id]['FAS_track_frame'])
						a1 = np.std(total_angle[:, 0])
						a2 = np.std(total_angle[:, 1])
						if a1 > 0.15 or a2 > 0.15:
							print('Real Face')
							self.people_tracked[id]['FAS'] = 1
						else:
							print('Fake Face')
							self.people_tracked[id]['FAS_track_frame'] = []
					else:
						self.people_tracked[id]['FAS_track_frame'].append(rotation_face.T[0])
				# face is real
				if self.people_tracked[id]['FAS'] == 1:
					# verify is not None: pass
					if self.people_tracked[id]['verify'] is None:
						if recog:
							feet = face_recognizer.face_encoding(frame, kps)
							info = face_compare_tree(feet, employees_data, search_tree)
							if info['Sim'] > self.verify_threshold:
								# insert_timekeeping(info['id'], info['fullname'], info['position'])
								if info['position'] == 'speaker':
									self.have_speaker = True
								self.people_tracked[id]['verify'] = info
		# print(self.people_tracked.keys())
		# print('-'*10)
			return self.people_tracked[id]['verify']
		else:
			return None

class TrackPartitipants():
	def __init__(self, img_size=(640, 480), time_track_fas=2, fps_process=10):
		self.people_tracked = {}
		self.frame_FAS = fps_process * time_track_fas
		self.frame_verify = 1
		self.verify_threshold = 0.7
		self.recog_frame = self.frame_verify

		# self.eye = Eye()

	def update(self, id=None, box=None, kps=None, frame=None, rotation=None):
		for i in self.people_tracked.keys():
			if datetime.now().second - self.people_tracked[i]['last_seen'].second >1: # no see 1s
				info = self.people_tracked[i]['verify']
				if info is not None:
					print('success')
					# insert_timekeeping(info['id'],
					# 				   info['fullname'],
					# 				   info['position'],
					# 				   self.people_tracked[i]['first_seen'].strftime("%Y-%m-%d %H:%M:%S"),
					# 				   self.people_tracked[i]['last_seen'].strftime("%Y-%m-%d %H:%M:%S"))
				self.people_tracked.pop(i)

		if id is not None:
			if not id in self.people_tracked.keys():
				self.people_tracked.update({id: {'box': box,
								  'kps': kps,
								  'FAS': 0,
								  'verify': None,
								  'FAS_track_frame':[],
								  'first_seen': datetime.now(),
								  'last_seen': datetime.now()}})
			else:
				# if box is None and kps is None:
				# 	self.people_tracked.pop(id)
				# 	return None
				# else: # Update old person
					# self.people_tracked[id] = [box, kps]
					# FAS method
				self.people_tracked[id]['last_seen'] = datetime.now()
				if self.people_tracked[id]['FAS'] == 0:
					if len(self.people_tracked[id]['FAS_track_frame']) >= self.frame_FAS:
						total_angle = np.array(self.people_tracked[id]['FAS_track_frame'])
						a1 = np.std(total_angle[:, 0])
						a2 = np.std(total_angle[:, 1])
						if a1 > 0.15 or a2 > 0.15:
							print('Real Face')
							self.people_tracked[id]['FAS'] = 1
						else:
							print('Fake Face')
							self.people_tracked[id]['FAS_track_frame'] = []
					else:
						self.people_tracked[id]['FAS_track_frame'].append(rotation_face.T[0])
				# face is real
				if self.people_tracked[id]['FAS'] == 1:
					# verify is not None: pass
					if self.people_tracked[id]['verify'] is None:
						if recog:
							feet = face_recognizer.face_encoding(frame, kps)
							info = face_compare_tree(feet, employees_data, search_tree)
							if info['Sim'] > self.verify_threshold:
								# insert_timekeeping(info['id'], info['fullname'], info['position'])
								self.people_tracked[id]['verify'] = info
		# print(self.people_tracked.keys())
		# print('-'*10)
			return self.people_tracked[id]['verify']
		else:
			return None




