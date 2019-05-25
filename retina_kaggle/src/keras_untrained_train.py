import keras;
import tensorflow as tf;
import os;
from math import ceil;

#Meta Start
model_name = "kaggle_retinopathy";
model_version = "_299_0.0";
#model_version = "_600_0.0";
#Meta End

#Paths start
training_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/training";
validation_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/validation";

model_dir = "../models/"

model_save_path = model_dir + model_name + model_version + ".h5"
#Paths end


#Model/Data parameters start
number_of_classes = 5;
classes = {
	0: "No_DR",
	1: "Mild_DR",
	2: "Moderate_DR",
	3: "Severe_DR",
	4: "Proliferative_DR"
};


input_shape = (299, 299, 3);
#input_shape = (600, 600, 3);
#pretrained_input_shape = (299, 299, 3);
#convolution_core = (2, 2);
#Model/Data parameters end


#Training Parameters Start
epochs = 32;
learning_rate = 1E-3;
batch_size = 8;
#Training Parameters End

#-------------------------------------------------------------------------

def init_model(number_of_classes = number_of_classes, learning_rate = learning_rate):
	
    model = keras.applications.inception_v3.InceptionV3(weights = None, classes = number_of_classes);
    optimizer = keras.optimizers.SGD(lr = learning_rate);
    loss = keras.losses.mean_squared_error;
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy']);

    return model;

def train_model(model, training_set_dir = training_set_dir, validation_set_dir = validation_set_dir,\
                          epochs = epochs, learning_rate = learning_rate, batch_size = batch_size):

    data_generator = keras.preprocessing.image.ImageDataGenerator\
    (
        rescale = 1. / 255
    );

    training_flow = data_generator.flow_from_directory\
    (
        directory = training_set_dir,
        target_size = (input_shape[0], input_shape[1]),
        class_mode = "categorical",
        batch_size = batch_size,
        interpolation = "bicubic"
    );

    validation_flow = data_generator.flow_from_directory\
    (
        directory = validation_set_dir,
        target_size = (input_shape[0], input_shape[1]),
        class_mode = "categorical",
        batch_size = batch_size,
        interpolation = "bicubic"
    );

    model.fit_generator\
    (
        generator = training_flow,
        steps_per_epoch = ceil( 28100 / batch_size ),
        epochs = epochs,
        verbose = 2,
        validation_data = validation_flow,
        validation_steps = ceil( 7026 / batch_size ),
        validation_freq = 8

    );

    return model;

def main():

    graph = tf.get_default_graph();

    with graph.as_default():

        model = train_model(init_model());
        model.save(model_save_path);

if __name__ == "__main__":

    main();
