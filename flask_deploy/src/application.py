"""
 take an image path and return a prediction
"""

import tensorflow as tf;
import os, random;
from PIL import Image;
import numpy as np;
import sys;

model_path = "../models/inceptionv3_retina_screening1.h5";

test_image_path = "../../../Datasets/STARE/training/normal/im0239.ppm";

if (len(sys.argv) < 2):
	path_to_the_image = test_image_path;
else:
	path_to_the_image = sys.argv[1];


input_shape = (600, 600, 3);

model = tf.keras.models.load_model(model_path);
#model.summary();

def screen(model = model, path_to_the_image = path_to_the_image):

	image = tf.keras.preprocessing.image.load_img\
	(
		path_to_the_image,
		target_size = (input_shape[0], input_shape[1])
	);

	tensor = tf.keras.preprocessing.image.img_to_array(image);

	input_tensor = tensor.reshape((1,) + tensor.shape);

	score = model.predict(input_tensor);

	inference = score * 1000 - 453;

	if (inference > 0.5):
		prediction = "Normal";
	else:
		prediction = "Abnormal";

	return prediction;

def main():

	result = screen(model = model, path_to_the_image = path_to_the_image);
	print(result);
	
	return result;

if __name__ == "__main__":
	main();
