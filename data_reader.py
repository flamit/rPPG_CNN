from torch.utils.data import Dataset
from skimage.measure import block_reduce
import numpy as np
import cv2
import os

# All videos are placed into image folders
# All gt are named the same as the folder
# Sample a folder name during training
# Sample frames
# load gt

# Input will be dir names

class FaceFrameReaderTrain(Dataset):

    def __init__(self, dir_paths, image_size, T=100, n=16):
        self.dir_paths = dir_paths
        self.image_names = [[x for x in os.listdir(y)] for y in dir_paths]
        self.image_size = image_size
        self.T = T
        self.max_idx = [len(x) - T for x in dir_paths]
        self.n = n
        self.count = 0
        for image_names in self.image_names:
            self.count += len(image_names)

    def read_gt_file(self, idx):
        gt_file = self.dir_paths[idx] + '.txt'
        with open(gt_file) as file:
            data = file.read()
        data = [float(x) for x in data.split() if float(x) > 60]

        return data

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        frames = []
        idx = np.random.randint(0, len(self.dir_paths))
        start_frame_idx = np.random.randint(0, len(self.image_names[idx]) - self.T)
        for i in range(start_frame_idx, start_frame_idx + self.T):
            path = os.path.join(self.dir_paths[idx], self.image_names[idx][i])
            image = cv2.imread(path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image = cv2.resize(image, self.image_size)
            image = block_reduce(image, (self.n, self.n, 1), np.mean)
            image = np.reshape(image, [1, -1, 3])
            frames.append(image)
        images_stacked = np.concatenate(frames)
        images_stacked = np.transpose(images_stacked, [2, 1, 0])
        gt = np.asarray(self.read_gt_file(idx)[start_frame_idx:start_frame_idx + self.T])
        return images_stacked, gt


class FaceFrameReaderTest(Dataset):

    def __init__(self, dir_paths, image_size, T=100, n=16):
        self.dir_paths = dir_paths
        self.image_names = [[x for x in os.listdir(y)] for y in dir_paths]
        self.image_size = image_size
        self.T = T
        self.max_idx = [len(x) - T for x in dir_paths]
        self.n = n
        self.count = 0
        for image_names in self.image_names:
            self.count += len(image_names)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        frames = []
        idx = np.random.randint(0, len(self.dir_paths))
        start_frame_idx = np.random.randint(0, len(self.image_names[idx]) - self.T)
        for i in range(start_frame_idx, start_frame_idx + self.T):
            path = os.path.join(self.dir_paths[idx], self.image_names[idx][i])
            image = cv2.imread(path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image = cv2.resize(image, self.image_size)
            image = block_reduce(image, (self.n, self.n, 1), np.mean)
            image = np.reshape(image, [1, -1, 3])
            frames.append(image)
        images_stacked = np.concatenate(frames)
        images_stacked = np.transpose(images_stacked, [2, 1, 0])

        return images_stacked
