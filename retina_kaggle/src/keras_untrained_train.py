import keras;
import tensorflow as tf;
import os;
from math import ceil;

#Meta Start
model_name = "kaggle_retinopathy";
model_version = "_299_0.4";
#ver 0.1 - changed activation relu to sigmoid and trained for 32 epochs instead of 7.
#ver 0.2 - unfreezed all the layers and model is not pretrained, batch size = 32, 128 epochs.
#ver 0.3 - changed activation back to relu; unfreezed all the layers and model is not pretrained, batch size = 32, 128 epochs.
#ver 0.4 - trained for 1000 epochs
#model_version = "_600_0.0";
#Meta End

#Paths start
training_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/training";
validation_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/validation";

model_dir = "../models/"

model_save_path = model_dir + model_name + model_version + ".h5"
model_checkpoint_path = model_dir + model_name + model_version + "checkpoint.h5"
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
epochs = 2#1000;
learning_rate = 1E-3;
batch_size = 32;
#Training Parameters End

#-------------------------------------------------------------------------

def init_model(number_of_classes = number_of_classes, learning_rate = learning_rate):
	
    inception_model = keras.applications.inception_v3.InceptionV3(weights = None);#, classes = number_of_classes);

    inception_model.layers.pop();

    # for layer in inception_model.layers:
    #     layer.trainable = False;

    new_model_output = keras.layers.Dense(number_of_classes, activation = "relu")(inception_model.layers[-1].output)
    model = keras.models.Model(inception_model.input, new_model_output);
    
    optimizer = keras.optimizers.SGD(lr = learning_rate);
    loss = keras.losses.mean_squared_error;
    model.compile(optimizer = optimizer, loss = loss, metrics = ["accuracy"]);

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
        batch_size = batch_size#,
        #interpolation = "bicubic"
    );

    validation_flow = data_generator.flow_from_directory\
    (
        directory = validation_set_dir,
        target_size = (input_shape[0], input_shape[1]),
        class_mode = "categorical",
        batch_size = batch_size#,
        #interpolation = "bicubic"
    );

    model_checkpoint = keras.callbacks.ModelCheckpoint(model_checkpoint_path, verbose = 1, monitor = 'val_acc', save_best_only = True, save_weights_only = False,
        mode = "max", period = 10);

    model.fit_generator\
    (
        generator = training_flow,
        steps_per_epoch = ceil( 28100 / batch_size ),
        epochs = epochs,
        verbose = 1,
        callbacks = [model_checkpoint],
        validation_data = validation_flow,
        validation_steps = ceil( 7026 / batch_size ),
        validation_freq = 10

    );

    return model;

def main():

    graph = tf.get_default_graph();

    with graph.as_default():

        model = train_model(init_model());

        if (not os.path.exists(model_dir)):
            os.mkdir(model_dir);
            
        model.save(model_save_path);

if __name__ == "__main__":

    main();