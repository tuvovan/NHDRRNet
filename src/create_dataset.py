import os
import sys
import cv2
import glob
import math
import numpy as np
import h5py
import shutil
import random
import tensorflow as tf

def populate_train_list(images_path, train=True):
    imgs_np = np.zeros([1, 4, 512, 512, 6])
    list_dir = sorted(os.listdir(images_path))
    for i, folder in enumerate(list_dir):
        print(i)
        if os.path.isdir(os.path.join(images_path, folder)):
            file1 = open(os.path.join(images_path, folder, 'exposure.txt'), 'r') 
            Lines = file1.readlines() 
            t = [float(k) for k in Lines]
            for m in range(2):
                for n in range(4):
                    a = 256*m
                    b = 256*n
                    print(i,'--->',a,b)
                    batch_np = np.zeros([4, 512, 512, 6])
                    batch_np_rt = np.zeros([4, 512, 512, 6])
                    list_file = sorted(glob.glob(os.path.join(images_path, folder, '*.tif')))
                    for j, f in enumerate(list_file):
                        ldr = (cv2.imread(f)).astype(np.float32)
                        ldr = ldr / 255.0
                        ldr = ldr[a:a+512, b:b+512]

                        hdr = ldr**2.2 / (2**t[j])

                        X = np.concatenate([ldr, hdr], axis=-1)
                        # X_rt = distortion(X)
                        X = np.expand_dims(X, axis=0)
                        # X_rt = np.expand_dims(X_rt, axis=0)
                        batch_np[j,:,:,:] = X
                        # batch_np_rt[j,:,:,:] = X_rt
            
                    for f in glob.glob(os.path.join(images_path, folder, '*.hdr')):
                        hdr = (cv2.imread(f, -1)).astype(np.float32)

                        hdr = hdr[a:a+512, b:b+512]
                        hdr_0 = np.zeros_like(hdr)
                        X_hdr = np.concatenate([hdr, hdr_0], axis=-1)
                        # X_hdr_rt = distortion(X_hdr)
                        X_hdr = np.expand_dims(X_hdr, axis=0)
                        # X_hdr_rt = np.expand_dims(X_hdr_rt, axis=0)
                        batch_np[3,:,:,:] = X_hdr
                        # batch_np_rt[3,:,:,:] = X_hdr_rt
                    batch_np_rt = distortion(batch_np)
                    batch_np = np.expand_dims(batch_np, axis=0)
                    batch_np_rt = np.expand_dims(batch_np_rt, axis=0)
                    imgs_np = np.append(imgs_np, batch_np, axis=0)
                    imgs_np = np.append(imgs_np, batch_np_rt, axis=0)

        data = imgs_np[1:,:,:,:,:]
        if train:
            np.save('data_01_512.npy', data)
        else:
            np.save('data_val_01.npy', data)
        print('saved %d images'%(i+1))

def distortion(imgs):
    distortions = tf.random.uniform([2], 0, 1.0, dtype=tf.float32)
    
    # flip horizontally
    imgs = tf.cond(tf.less(distortions[0],0.5), lambda: tf.image.flip_left_right(imgs), lambda: imgs)
    
    # rotate
    k = tf.cast(distortions[1]*4+0.5, tf.int32)
    imgs = tf.image.rot90(imgs, k)

    return imgs

populate_train_list('/home/inhand/Tu/AHDR/Training/', True)