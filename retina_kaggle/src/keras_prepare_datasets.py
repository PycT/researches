import os, shutil;
from math import floor;
from tqdm import tqdm;

#Meta Start
#model_version = "_600_0.0";
model_version = "_299_0.0";
#Meta End

#Paths start
dataset_img_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/train";
training_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/training";
validation_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/validation";
dataset_labels_path = "/home/pyct/DATA/Datasets/kaggle_retinopathy/labels.csv";
model_dir = "models/"
model_name = "kaggle_retinopathy"
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
pretrained_input_shape = (299, 299, 3);
convolution_core = (2, 2);
#Model/Data parameters end

#--------------------------------------------------------------------------------------------------------------------------------------

def prepare_dir_structure(training_set_dir = training_set_dir,\
                  validation_set_dir = validation_set_dir):
    
    if (not os.path.exists(validation_set_dir)):
        os.mkdir(validation_set_dir);
    
    if (not os.path.exists(training_set_dir)):
        os.mkdir(training_set_dir);
        
    for key in classes:
        
        if (not os.path.exists(training_set_dir + "/" + classes[key])):
            os.mkdir(training_set_dir + "/" + classes[key])
        
        if (not os.path.exists(validation_set_dir + "/" + classes[key])):
            os.mkdir(validation_set_dir + "/" + classes[key])
        
    return True;

def prepare_sets(sets_split = 0.8, labels_path = dataset_labels_path, source_img_dir = dataset_img_dir,\
    training_set_dir = training_set_dir, validation_set_dir = validation_set_dir):
    
    labels_file = open(labels_path, "r");
    labels = labels_file.readlines();
    labels_file.close();

    training_set_size = floor(len(labels) * sets_split);

    i = 1;

    target_dir = training_set_dir;

    for label in tqdm(labels):

        item = label.strip().split(",");

        if (i > training_set_size) and (target_dir == training_set_dir):
            target_dir = validation_set_dir;

        source_img = "{}/{}.jpeg".format(source_img_dir, item[0]);
        destination_img = "{}/{}/{}.jpeg".format(target_dir, classes[int(item[1])], item[0]);

        shutil.copy(source_img, destination_img);

        i += 1;

    return True;

def main():

    prepare_dir_structure();
    prepare_sets();

if __name__ == "__main__":
    main();