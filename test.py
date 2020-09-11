import os
import sys
import HDR
import glob
import time
import math
import argparse
import numpy as np
import tensorflow as tf 
import generate_HDR_dataset

from val import run
from PIL import Image
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Concatenate, Conv2D, Input


def get_test_data_real(images_path):
    imgs_np = np.zeros([1, 3, 768, 1024, 6])
    file1 = open(os.path.join(images_path, 'exposure.txt'), 'r') 
    Lines = file1.readlines() 
    t = [float(i) for i in Lines]

    for j, f in enumerate(sorted(glob.glob(os.path.join(images_path, '*.tif')))):
        ldr = (cv2.imread(f, -1)/65535.0).astype(np.float32)
        ldr = cv2.resize(ldr, (1024,768))
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        hdr = ldr**2.2 / (2**t[j])
        X = np.concatenate([ldr, hdr], axis=-1)
        imgs_np[0,j,:,:,:] = X

    return imgs_np


def run(config, model):
    SDR = get_test_data_real(config.test_path)
    rs = model.predict(SDR)
    out = rs[0]
    tonemap = cv2.createTonemapReinhard()
    out = tonemap.process(out.copy())
    cv2.imwrite(os.path.join(config.test_path, 'hdr.jpg'), np.uint8(out*255))



if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--test_path', type=str, default="Test/EXTRA/001/")
	parser.add_argument('--gpu', type=int, default=1)
	parser.add_argument('--weight_test_path', type=str, default= "weights/epoch_rgb.h5")
	parser.add_argument('--filter', type=int, default= 32)
	parser.add_argument('--kernel', type=int, default= 3)
	parser.add_argument('--encoder_kernel', type=int, default= 3)
	parser.add_argument('--decoder_kernel', type=int, default= 4)
	parser.add_argument('--triple_pass_filter', type=int, default= 256)

	config = parser.parse_args()

	if not os.path.exists(config.checkpoints_folder):
		os.mkdir(config.checkpoints_folder)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

	model_x = NHDRRNet(config)
    x = Input(shape=(3, 256, 256, 6))
    out = model_x.main_model(x)
    model = Model(inputs=x, outputs=out)
    model.load_weights(config.weight_test_path)
    model.summary()

	run(config, model)
