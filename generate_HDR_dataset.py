import os
import sys
import cv2
import glob
import math
import numpy as np
import h5py
import shutil
import random

def load_data(npy_path):
    return np.load(npy_path)

class DataGenerator():
    def __init__(self, images_path, batch_size):
        self.shuffle = True
        self.imgs= load_data(images_path) 
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.imgs) / self.batch_size)

    def __getitem__(self, idx):
        self.indices = np.arange(self.imgs.shape[0]).astype(np.uint32)
        np.random.shuffle(self.indices)
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs_x = self.imgs[inds]
        return imgs_x

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(self.imgs.shape[0]).astype(np.uint32)
        if self.shuffle == True:
            np.random.shuffle(self.indices)
