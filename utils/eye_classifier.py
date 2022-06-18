# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import torch
from torch import nn
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from utils.face_aligner import get_similarity_transform


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(2304, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# if __name__ == '__main__':
#     x = torch.rand((1, 1, 32, 32))
#     net = CNN()
#     out = net(x)


class Eye():

    def __init__(self, model_file='src/eye_blink/best.h5', device='cpu'):
        self.device = device
        self.load(model_file)
        self.eye.eval()

    def load(self, model):
        self.eye = CNN()
        self.eye.load_state_dict(torch.load(model, map_location=self.device))

    def predict(self, frame, kps):
        left_eye, right_eye = self.crop_left_eye(frame, kps)
        left_eye_batch = self.preprocess_image(left_eye, right_eye)
        out = self.eye(left_eye_batch.to(self.device))
        _, index = torch.max(out, 1)
        pred = index.cpu().numpy().tolist()
        pred = pred[0] + pred[1]
        return pred, left_eye, right_eye

    def preprocess_image(self, left_eye, right_eye):
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(left_eye)
        # b, g, r = img.split()
        # img = Image.merge("RGB", (r, g, b))
        # img = Image.open(i).convert('RGB')
        input_size = 32
        normalize = transforms.Normalize(mean=[0.5],
                                         std=[0.5])
        preprocess = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize])
        left_eye_preprocessed = preprocess(img)

        img = Image.fromarray(right_eye)
        # b, g, r = img.split()
        # img = Image.merge("RGB", (r, g, b))
        right_eye_preprocessed = preprocess(img)

        batch_img_tensor = torch.stack((left_eye_preprocessed, right_eye_preprocessed), 0)
        return batch_img_tensor

    def crop_left_eye(self, frame, kps):
        image_size = 112
        M = get_similarity_transform(kps)
        warped = cv2.warpAffine(frame, M, (image_size, image_size), borderValue=0.0)
        # New kps calc
        ones = np.ones(shape=(len(kps), 1))
        points_ones = np.hstack([kps, ones])
        new_kps = M.dot(points_ones.T).T
        #crop left eye
        eye_shape = int(np.linalg.norm(new_kps[0] - new_kps[1])/3)

        centroid_left_eye = new_kps[0].astype(np.int)
        left_eye = warped[centroid_left_eye[1]-eye_shape:centroid_left_eye[1]+eye_shape, centroid_left_eye[0]-eye_shape:centroid_left_eye[0]+eye_shape, :]
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        # left_eye = np.stack((left_eye,) * 3, axis=-1)
        # print(left_eye.shape)

        centroid_right_eye = new_kps[1].astype(np.int)
        right_eye = warped[(centroid_right_eye[1]-eye_shape):(centroid_right_eye[1]+eye_shape), centroid_right_eye[0]-eye_shape:centroid_right_eye[0]+eye_shape, :]
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        # right_eye = np.stack((right_eye,) * 3, axis=-1)

        return left_eye, right_eye

