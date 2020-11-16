import os
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms
from sklearn.model_selection import train_test_split
import csv

def load_class_mappings(filename='class_list.csv'):
    assert os.path.exists(filename), 'folder: {} not found.'.format(filename)
    with open(filename,'r') as f:
        reader = csv.reader(f)
        class_list = []
        for values in reader:
            if values[0] == 'index':
                continue
            class_list.append(values)
        class_mappings = ['']*len(class_list)
        for cls in class_list:
            class_mappings[int(cls[0])] = cls[1]
        return class_mappings
            

class LoadDatasetFromFolder():
    def __init__(self,folder_name=None,class_list_file=None):
        self.folder_name = folder_name
        assert os.path.exists(self.folder_name), 'folder: {} not found.'.format(self.folder_name)
        print("Loading dataset from folder {0}".format(str(Path(folder_name))))
        self.class_mappings = load_class_mappings(class_list_file)
        for folder in glob.glob(folder_name+'/*/*'):
            self.class_list.append(os.path.basename(folder))
        self.X = []
        self.y = []
        for folder in glob.glob(self.folder_name+'/*/*'):
            class_label = os.path.basename(folder)
            #label_index = self.class_list.index(class_label)
            label = self.class_list.index(class_label)
            for image in glob.glob(folder+'/*.jpg'):
                with Image.open(image).convert('RGB') as img:
                    img = np.array(img)
                    if len(img.shape) > 3:
                        print(image)
                    self.X.append(img)
                    #label = np.zeros(len(self.class_list))
                    #label[label_index] = 1
                    self.y.append(label)
        print("Dataset Loaded successfully")
        
    def load(self):
        return np.array(self.X),np.array(self.y)

    def get_classes_list(self):
        return self.class_list

    def num_classes(self):
        return len(self.class_list)

    def get_class_mappings(self):
        return self.class_mappings



class FaceDataset(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])

    def __init__(self,images,labels,train=False,transform=False):
        self.images = images
        self.labels = labels
        self._transform = transform
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img, target = self.images[index], self.labels[index]
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = torchvision.transforms.Resize(224)(img)
        if self.train:
            img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
        if self._transform:
            img = self.transform(img)
        if len(img.shape) > 3:
            print(image)
        return img,target

    def transform(self,img):
        img = np.array(img,dtype=np.uint8)
        img = img[:,:,::-1]
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2,0,1) #C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    def untransform(self,img,lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img.astype(np.uint8)
        img = img[:,:,::-1]
        return img,lbl

class CreateTrainValDatasets():
    def __init__(self,X,y,val_size=0.12,random_state=0,stratify=True):
        print("Creating Train and Val datasets")
        self.X = X
        self.y = y
        self.val_size=val_size
        self.random_state=random_state
        self.stratify = stratify

        if self.stratify:
           self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X,y,test_size=val_size,random_state=random_state,stratify=y)
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X,y,test_size=val_size,random_state=random_state)
        
        self.trainset = FaceDataset(self.X_train, self.y_train, train=True,transform=True)
        self.valset = FaceDataset(self.X_val,self.y_val,transform=True)
        print("Train and Val datasets created successfully")

    def get_trainset(self):
        return self.trainset
    
    def get_valset(self):
        return self.valset

if __name__ == "__main__":
    pass


