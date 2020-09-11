import tensorflow as tf 
import os
import sys
import argparse 
import time
import prepare_data
import numpy as np
import glob
import math
import cv2

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D, Input
from PIL import Image
import generate_HDR_dataset
from model import *
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



def get_test_data_real(images_path):
    imgs_np = np.zeros([1, 3, 768, 1024, 6])
    file1 = open(os.path.join(images_path, 'exposure.txt'), 'r') 
    Lines = file1.readlines() 
    t = [float(i) for i in Lines]

    for j, f in enumerate(sorted(glob.glob(os.path.join(images_path, '*.tif')))):
        ldr = (cv2.imread(f)/255.0).astype(np.float32)
        ldr = cv2.resize(ldr, (1024,768))
        hdr = ldr**2.2 / (2**t[j])
        # ldr = ldr *2.0 - 1
        # hdr = hdr * 2.0 - 1
        X = np.concatenate([ldr, hdr], axis=-1)
        imgs_np[0,j,:,:,:] = X

    return imgs_np


def run(config, model):
# def run(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    MU = 5000.0
    SDR = get_test_data(config.test_path)


    # model_x = AHDR(config)
    # x = Input(shape=(3, 500, 750, 6))
    # out = model_x.main_model(x)

    # model = Model(inputs=x, outputs=out)
    # model.load_weights('/home/inhand/Tu/AHDR/weights/best.h5')

    rs = model.predict(SDR)
    out = rs[0]
    out = tf.math.log(1 + MU * out) / tf.math.log(1 + MU)
    # out = (tf.math.exp(out[0] * tf.math.log(1.0 + 5000.0))-1.0)/5000.0
    # out = (rs[0]) ** (1/2.0)
    # out = tf.math.log(1.0 + 5000.0*rs[0])/tf.math.log(1.0 + 5000.0)

    cv2.imwrite(os.path.join(config.test_path, 'hdr.jpg'), np.uint8(out*255))



if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--images_path', type=str, default="/home/inhand/Tu/AHDR/Training/")
	parser.add_argument('--test_path', type=str, default="/home/inhand/Tu/AHDR/Test/EXTRA/009/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--gpu', type=int, default=1)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=20)
	parser.add_argument('--checkpoint_iter', type=int, default=1)
	parser.add_argument('--checkpoints_folder', type=str, default="weights/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "weights/Epoch10.h5")
	parser.add_argument('--filter', type=int, default= 64)
	parser.add_argument('--kernel', type=int, default= 3)
	parser.add_argument('--growth_rate', type=int, default= 32)

	config = parser.parse_args()

	if not os.path.exists(config.checkpoints_folder):
		os.mkdir(config.checkpoints_folder)

	# x = Input(shape=(3, 768, 1024, 6))
	# out = HDR.main_model(x)

	# model = Model(inputs=x, outputs=out)
	# model.load_weights('/home/inhand/Tu/AHDR/weights/best.h5')
	# run(config, model)
