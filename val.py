import tensorflow as tf 
import os
import sys
import argparse 
import time
import numpy as np
import glob
import math
import cv2

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D, Input
from PIL import Image
import generate_HDR_dataset
import HDR

def get_test_data(images_path):
    imgs_np = np.zeros([1, 3, 256, 256, 6])
    file1 = open(os.path.join(images_path, 'exposure.txt'), 'r') 
    Lines = file1.readlines() 
    t = [float(i) for i in Lines]

    for j, f in enumerate(sorted(glob.glob(os.path.join(images_path, '*.tif')))):
        ldr = (cv2.imread(f)/255.0).astype(np.float32)
        ldr = cv2.resize(ldr, (256,256))
        hdr = ldr**2.2 / (2**t[j])
        # ldr = ldr *2.0 - 1
        # hdr = hdr * 2.0 - 1
        X = np.concatenate([ldr, hdr], axis=-1)
        imgs_np[0,j,:,:,:] = X

    return imgs_np


def run(config, model):
    MU = 5000.0
    SDR = get_test_data(config.test_path)

    rs = model.predict(SDR)
    out = rs[0]
    out = tf.math.log(1 + MU * out) / tf.math.log(1 + MU)

    cv2.imwrite(os.path.join(config.test_path, 'hdr.jpg'), np.uint8(out*255))
