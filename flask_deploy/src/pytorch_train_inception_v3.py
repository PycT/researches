import os, shutil;
import math, random;

from torch import nn;
import torch.optim;
from torch.utils.data import DataLoader, Dataset;
from torchvision import models, transforms;
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader;
from torchvision.datasets import ImageFolder;

import numpy as np;
from tqdm import tqdm;

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



class InceptionModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(InceptionModel, self).__init__()
        model = models.inception_v3(pretrained=pretrained)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
        model.fc = nn.Linear(2048, num_classes)
        #model.out = nn.Sigmoid()
        self.model = model

    def forward(self, x):
        return self.model(x)
    


model = InceptionModel(prediction_classes_quantity, True);
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001);



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

#prepare datasets functions

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

def train_the_model(model, batch_size, patience, training_epochs):
    """
    'patience' is a number of interations without loss improvement, after which training is stopped.
    """    
    early_stopping = EarlyStopping(patience = patience, verbose = True);
    
    data_transform = transforms.Compose\
    (
        [
            transforms.Resize((299, 299)),
            #transforms.RandomHorizontalFlip();
            #transforms.RandomRotation(180),
            transforms.ToTensor()#,
            #transforms.Normalize\
            #(
            #    mean = [0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225]
            #)
        ]
    );
    
    training_dataset = ImageFolder(training_set_dir, transform = data_transform);
    training_data_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True);

    validation_dataset = ImageFolder(validation_set_dir, transform = data_transform);
    validation_data_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True);

    avg_training_loss_tracker = [];
    avg_validation_loss_tracker = [];

    for epoch in tqdm(range(1, training_epochs + 1)):

        training_loss_tracker = [];
        validation_loss_tracker = [];
        
        """training block starts"""
        model.train(); #initialize model for training;
        
        for batch, (samples, labels) in tqdm(enumerate(training_data_loader, 1), \
                                           desc = "Training", total = len(training_data_loader)):
            
            labels = torch.LongTensor(labels);
            
            outputs, outputs_aux = model(samples);
              
            loss1 = criterion(outputs, labels);
            loss2 = criterion(outputs_aux, labels);
            loss = loss1 + 0.4 * loss2;

            loss.backward();
            optimizer.step();

            training_loss_tracker.append(float(loss.item()));
            
            
            if batch % 10 == 0:
                tqdm.write("Train loss: {}".format(sum(training_loss_tracker) / len(training_loss_tracker)));
            
            del samples;
            del labels;
            
        """training block ends"""
                                           
        """validation block starts"""
        model.eval(); #initialize model for validation;
        
        for batch, (samples, labels) in tqdm(enumerate(validation_data_loader, 1), \
                                           desc = "Validation", total = len(validation_data_loader)):
            
            labels = torch.LongTensor(labels);
            
            outputs = model(samples);
            loss = criterion(outputs, labels);
            
            _, predictions = torch.max(outputs, 1);
            
            validation_loss_tracker.append(loss.item());
            
            running_accuracy = torch.sum(predictions == labels.data);

            training_loss_tracker.append(float(loss.item()));
            
            if batch % 5 == 0:
                tqdm.write("Validity {}".format(float(running_accuracy) / batch_size));
            
            del samples;
            del labels;
            
        """validation block ends"""
        
        avg_training_loss = np.average(training_loss_tracker);
        avg_validation_loss = np.average(validation_loss_tracker);
        
        avg_training_loss_tracker.append(avg_training_loss);
        avg_validation_loss_tracker.append(avg_validation_loss);
        
        early_stopping(avg_validation_loss, model);
        if early_stopping.early_stop:
            tqdm.write("Early stop.");
            break;
            
    return model, avg_training_loss_tracker, avg_validation_loss_tracker;

def main():

	#dir_structure();
	#whole_set = make_data_sets_lists(sets_chopper = 0.7);
	#fill_sets_dir(whole_set);

	mdl = train_the_model(model, batch_size = 3, patience = 8, training_epochs = 77);

	torch.save(mdl, "../models/retina_screening.pt");

	return True;

if __name__ == "__main__":

	main();