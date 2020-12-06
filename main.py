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
from HDR import *
from PIL import Image
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D, Input

def progress(epoch, trained_sample ,total_sample, bar_length=25, total_loss=0, message=""):
    percent = float(trained_sample) / total_sample
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rEpoch {0}, Iteration: {1}: [{2}] {3}%  ----- Loss: {4}".format(epoch, trained_sample, hashes + spaces, int(round(percent * 100)), float(total_loss)) + message)
    sys.stdout.flush()

def augment(data):
    mode = np.random.randint(0, 3)
    if mode == 0:
        return np.fliplr(data)
    elif mode == 1:
        return np.flipud(data)
    elif mode == 2:
        return np.rot90(data)
    else:
        return np.rot90(np.rot90(data))

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    MU = 5000.0
    SDR = generate_HDR_dataset.DataGenerator(config.images_path, config.train_batch_size)
    lr = config.lr

    model_x = NHDRRNet(config)
    x = Input(shape=(3, 512, 512, 6))
    out = model_x.main_model(x)
    model = Model(inputs=x, outputs=out)
    model.summary()

    if config.load_pretrain:
        model.load_weights(config.pretrain_dir)

    min_loss = 10000100
    print("Start training ...")
    for epoch in range(config.num_epochs):
        total_loss = 0
        if epoch+1 > 80000:
            if epoch+1 % 20000 == 0:
                lr = lr*0.9
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
        for iteration in range(len(SDR)):
            with tf.GradientTape() as tape:
                img_lowlight = SDR[iteration]
                img_lowlight = augment(img_lowlight)
                imgs = img_lowlight[:,:3,:,:,:]
                imgs = tf.dtypes.cast(imgs,tf.float32)
                gt = img_lowlight[:,3,:,:,:3]
                gt = tf.dtypes.cast(gt,tf.float32)
                out = model(imgs)
                
                gt = tf.math.log(1 + MU * gt) / tf.math.log(1 + MU)
                out = tf.math.log(1 + MU * out) / tf.math.log(1 + MU)
                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(gt, out)

            grads = tape.gradient(loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if (iteration+1) % config.checkpoint_ep == 0:
                message = ''
                if loss < min_loss:
                    min_loss = loss.numpy()
                    model.save_weights(os.path.join(config.checkpoints_folder, "best.h5"))
                    print(' min loss: %.5f'%min_loss)
            progress(epoch+1, (iteration+1), len(SDR), total_loss=loss, message='')

        if (epoch+1) % config.display_ep == 0:
            run(config, model)
        print('  --  evaluated, check results please!')
                




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters

	parser.add_argument('--images_path', type=str, default="Training/data_01.npy")
	parser.add_argument('--test_path', type=str, default="Training/010/")
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=160000)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--display_ep', type=int, default=1)
	parser.add_argument('--checkpoint_ep', type=int, default=1)
	parser.add_argument('--checkpoints_folder', type=str, default="weights/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "weights/best.h5")
	parser.add_argument('--filter', type=int, default= 32)
	parser.add_argument('--kernel', type=int, default= 3)
	parser.add_argument('--encoder_kernel', type=int, default= 3)
	parser.add_argument('--decoder_kernel', type=int, default= 4)
	parser.add_argument('--triple_pass_filter', type=int, default= 256)


	config = parser.parse_args()

	if not os.path.exists(config.checkpoints_folder):
		os.mkdir(config.checkpoints_folder)

	train(config)
