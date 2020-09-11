import os
import sys
import cv2
import glob
import math
import numpy as np
import h5py
import shutil
import random
# from torch.utils.data import Dataset, DataLoader, random_split

def populate_train_list(images_path):
    imgs_np = np.zeros([1, 4, 256, 256, 6])
    # a = random.randint(256, 1500)
    # b = random.randint(256,1000)
    # batch_np = np.zeros([4, 256, 256, 6])
    for i, folder in enumerate(os.listdir(images_path)):
        # a = random.randint(256, 1000)
        # b = random.randint(256,1500)
        # print(a,b)
        if os.path.isdir(os.path.join(images_path, folder)):
            file1 = open(os.path.join(images_path, folder, 'exposure.txt'), 'r') 
            Lines = file1.readlines() 
            t = [float(k) for k in Lines]
            # for m in range(3):
            #     for n in range(5):
            #         a = 256*m
            #         b = 256*n
                    # print(i,'--->',a,b)
            batch_np = np.zeros([4, 256, 256, 6])
            for j, f in enumerate(glob.glob(os.path.join(images_path, folder, '*.tif'))):
                ldr = (cv2.imread(f)/255.0).astype(np.float32)
                ldr = cv2.resize(ldr, (256,256))
                hdr = ldr**2.2 / (2**t[j])
                ldr = ldr *2.0 - 1
                hdr = hdr * 2.0 - 1
                X = np.concatenate([ldr, hdr], axis=-1)
                X = np.expand_dims(X, axis=0)
                batch_np[j,:,:,:] = X
    
            for f in glob.glob(os.path.join(images_path, folder, '*.hdr')):
                hdr = (cv2.imread(f, -1)).astype(np.float32)
                hdr = hdr * 2.0 -1
                hdr = cv2.resize(hdr, (256,256))
                hdr_0 = np.zeros_like(hdr)
                X_hdr = np.concatenate([hdr, hdr_0], axis=-1)
                X_hdr = np.expand_dims(X_hdr, axis=0)
                batch_np[3,:,:,:] = X_hdr
            batch_np = np.expand_dims(batch_np, axis=0)
            imgs_np = np.append(imgs_np, batch_np, axis=0)

    data = imgs_np[1:,:,:,:,:]
    np.save('data.npy', data)
    print('saved!')

def populate_train_list_full(images_path):
    imgs_np = np.zeros([1, 4, 500, 750, 6])
    for i, folder in enumerate(os.listdir(images_path)):
        print(i+1)
        if os.path.isdir(os.path.join(images_path, folder)):
            file1 = open(os.path.join(images_path, folder, 'exposure.txt'), 'r') 
            Lines = file1.readlines() 
            t = [float(k) for k in Lines]
            if t[0] == 0.0:
                t[0] = 0.0005 
            batch_np = np.zeros([4, 500, 750, 6])

            for j, f in enumerate(glob.glob(os.path.join(images_path, folder, '*.tif'))):
                ldr = (cv2.imread(f, -1)/65535).astype(np.float32)
                ldr = cv2.resize(ldr, (750, 500))
                hdr = ldr ** 2.2 / t[j]
                X = np.concatenate([ldr, hdr], axis=-1)
                X = np.expand_dims(X, axis=0)
                batch_np[j,:,:,:] = X
    
            for f in glob.glob(os.path.join(images_path, folder, '*.hdr')):
                hdr = (cv2.imread(f, -1)).astype(np.float32)
                hdr = cv2.resize(hdr, (750, 500))
                hdr_0 = np.zeros_like(hdr)
                X_hdr = np.concatenate([hdr, hdr_0], axis=-1)
                X_hdr = np.expand_dims(X_hdr, axis=0)
                batch_np[3,:,:,:] = X_hdr
            batch_np = np.expand_dims(batch_np, axis=0)
            imgs_np = np.append(imgs_np, batch_np, axis=0)

    data = imgs_np[1:,:,:,:,:]
    np.save('data_half.npy', data)
    print('saved!')

def load_data(npy_path):
    return np.load(npy_path)



class DataGenerator():

    def __init__(self, images_path, batch_size):
        self.shuffle = True
        self.imgs= load_data(images_path) 
        # self.imgs= populate_train_list(images_path)  
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