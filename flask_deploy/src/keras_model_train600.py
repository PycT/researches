import warnings;
warnings.filterwarnings("ignore");

import tensorflow as tf;
import os, shutil;
import math, random;

datasets_dir = "../../../Datasets/STARE";

initial_dir = datasets_dir + "/all-images";

training_set_dir = datasets_dir + "/training";
validation_set_dir = datasets_dir + "/validation";
classes = \
[
    "normal",
    "abnormal"
];

labels_file_path = datasets_dir + "/all-mg-codes.txt"

prediction_classes_quantity = 2; # "normal", "abnormal"

#model_path = "../models/nasnet_retina_screening.h5";
#model_path = "../models/nasnet_retina_screening600.h5";
model_path = "../models/inceptionv3_retina_screening600.h5";

input_shape = (600, 600, 3);
input_convolution_core = (2, 2);
#pretrained_input_shape = (331, 331, 3);
pretrained_input_shape = (299, 299, 3);

#==================================================================================================================
def dir_structure(datasets_dir = datasets_dir, training_set_dir = training_set_dir,\
                  validation_set_dir = validation_set_dir):
    
    if (not os.path.exists(validation_set_dir)):
        os.mkdir(validation_set_dir);
    
    if (not os.path.exists(training_set_dir)):
        os.mkdir(training_set_dir);
        
    for a_class in classes:
        
        if (not os.path.exists(training_set_dir + "/" + a_class)):
            os.mkdir(training_set_dir + "/" + a_class)
        
        if (not os.path.exists(validation_set_dir + "/" + a_class)):
            os.mkdir(validation_set_dir + "/" + a_class)
        
    return True;

def make_data_sets_lists(sets_chopper = 0.8, datasets_dir = datasets_dir, labels_file_path = labels_file_path):
    """
    This method is specific to STARE dataset.
    Sets_chopper sets the proportion between training and validation sets.
    """

    labels_file = open(labels_file_path, "r");
    
    normal_set = [];
    abnormal_set = [];
    
    for line in labels_file.readlines():
        raw = line.split();
        
        if (int(raw[1]) == 0):
            normal_set.append({"file": raw[0], "label": 0});
        else:
            abnormal_set.append({"file": raw[0], "label": 1});
                
    random.shuffle(normal_set);
    random.shuffle(abnormal_set);

    abnormal_set = abnormal_set[:len(normal_set)];# !!!!!!!!!!!!!!!!!!!!!!!!!cutting abnormal set!
    
    normal_set_size = len(normal_set);
    abnormal_set_size = len(abnormal_set);
    
    training_normal_set_size = math.floor(normal_set_size * sets_chopper);
    validation_normal_set_size = normal_set_size - training_normal_set_size;
    
    training_abnormal_set_size = math.floor(abnormal_set_size * sets_chopper);
    validation_abnormal_set_size = abnormal_set_size - training_abnormal_set_size;
    
    training_set = normal_set[:training_normal_set_size] + abnormal_set[:training_abnormal_set_size];
    validation_set = normal_set[training_normal_set_size:] + abnormal_set[training_abnormal_set_size:];
        
    return {"training_set": training_set, "validation_set": validation_set};

def fill_sets_dir(whole_set):
    """
    copy files from common folder to training/validation classes folder
    """
    
    for sample in whole_set["training_set"]:
        shutil.copy(initial_dir + "/" + sample["file"] + ".ppm", training_set_dir + "/" + classes[sample["label"]]);

    for sample in whole_set["validation_set"]:
        shutil.copy(initial_dir + "/" + sample["file"] + ".ppm", validation_set_dir + "/" + classes[sample["label"]]);
    
    return True;
#==================================================================================================================

def train_the_model(model, batch_size = 3, training_epochs = 77):

	data_generator = tf.keras.preprocessing.image.ImageDataGenerator\
	(
		rescale = 1. / 255,
		featurewise_center = True,
		featurewise_std_normalization = True
	);

	training_set_generator = data_generator.flow_from_directory\
	(
		training_set_dir,
		target_size = (input_shape[0], input_shape[1]),
		batch_size = batch_size,
		class_mode = "binary"
	);

	validation_set_generator = data_generator.flow_from_directory\
	(
		validation_set_dir,
		target_size = (input_shape[0], input_shape[1]),
		batch_size = batch_size,
		class_mode = "binary"
	);

	model.fit_generator\
	(
		training_set_generator,
		epochs = training_epochs,
		steps_per_epoch = math.floor(165 / batch_size),
		#validation_data = validation_set_generator,
		#validation_steps = math.floor(64 / batch_size),
		#class_weight = {0: 0.1, 1: 0.9},
		verbose = 1
	);

	return model;

def prepare_dataset():

	dir_structure();
	whole_set = make_data_sets_lists(sets_chopper = 0.7);
	fill_sets_dir(whole_set);

	return True;

def main():

	#pretrained_model = tf.keras.applications.resnet50.ResNet50(include_top = True, weights = None, classes = prediction_classes_quantity);
	#pretrained_model = tf.keras.applications.nasnet.NASNetLarge(include_top = True, weights = "imagenet");
	pretrained_model = tf.keras.applications.inception_v3.InceptionV3(include_top = True, weights = None);

	for layer in pretrained_model.layers[:-1]:
		layer.trainable = False;

	pretrained_model.summary();

	model = tf.keras.models.Sequential();
	model.add(tf.keras.layers.Conv2D(3, input_convolution_core, input_shape = input_shape));
	model.add(tf.keras.layers.MaxPooling2D());
	#model.add(tf.keras.layers.Reshape(pretrained_input_shape));
	model.add(pretrained_model);
	model.add(tf.keras.layers.Dense(1));
	model.add(tf.keras.layers.Activation('sigmoid'));

	model.summary();

	model.compile\
	(
		loss = "binary_crossentropy",
		optimizer = "adam",
		metrics = \
		[
			"accuracy"
		]
	);
	
	model = train_the_model(model, batch_size = 12, training_epochs = 62);

	model.save(model_path);

	print(model_path);

	return True;

if __name__ == "__main__":
	main();







