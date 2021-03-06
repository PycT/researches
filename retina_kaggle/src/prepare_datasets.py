import os, shutil;
from math import floor;
from tqdm import tqdm;
from PIL import Image;

#Meta Start

#Paths start
dataset_img_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/train";
training_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/training";
validation_set_dir = "/home/pyct/DATA/Datasets/kaggle_retinopathy/validation";
dataset_labels_path = "/home/pyct/DATA/Datasets/kaggle_retinopathy/labels.csv";
#Paths end


#Model/Data parameters start
number_of_classes = 5;
classes = {
    0: "0_No_DR",
    1: "1_Mild_DR",
    2: "2_Moderate_DR",
    3: "3_Severe_DR",
    4: "4_Proliferative_DR"
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

        source_img_path = "{}/{}.jpeg".format(source_img_dir, item[0]);
        destination_img_path = "{}/{}/{}.jpeg".format(target_dir, classes[int(item[1])], item[0]);

        the_image = Image.open(source_img_path);
        resized_image = the_image.resize((input_shape[0], input_shape[1]), Image.LANCZOS);
        resized_image.save(destination_img_path);
        #shutil.copy(source_img, destination_img);

        i += 1;

    return True;

def main():

    prepare_dir_structure();
    prepare_sets();

if __name__ == "__main__":
    main();