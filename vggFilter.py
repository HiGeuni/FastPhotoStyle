import argparse
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import cv2
import numpy as np
import json
import glob
import re

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def image_preprocess(file_path):
	img = load_img(file_path,target_size=(224, 224))
	img = img_to_array(img)
	tmp = np.expand_dims(img, axis=0)
	tmp = preprocess_input(tmp)
	return tmp

def getVggFeatures(file_path):
        tmp = image_preprocess(file_path)
        features = model.predict(img)
        return features
        
#define pretrained vgg19 model
model = VGG19(weights='imagenet')
model = Model(inputs = model.input, outputs=model.get_layer('block5_pool').output)

json_data = {}

dir_path = './images/**/*'
file_list = glob.glob(dir_path)
for file_path in file_list:
	if "result" in file_path:
		continue
	img = image_preprocess(file_path)
	features = model.predict(img)
	json_data[file_path] = features.tolist()
	print('complete', file_path)

with open('./data.json', 'w') as f:
	json.dump(json_data, f)
