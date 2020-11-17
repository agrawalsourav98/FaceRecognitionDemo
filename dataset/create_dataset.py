import glob
from PIL import ImageOps, Image
from pathlib import Path
import shutil
import argparse
import os
import torch
import cv2
import sys
import time
import numpy as np
import warnings
import csv

from datetime import timedelta

from mtcnn.mtcnn import MTCNN

warnings.filterwarnings("error")

def create_class_mappings(folder_name=None,path='class_list.csv'):
    """
    This function is used to create class mappings to indexes and save them to a csv file

    Attributes
    ----------
    folder_name : str
        The name of the folder which will be read to generate the class mappings
    path : str
        The path where the mappings are saved
    """
    print('\nCreating class mappings and saving...')
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    class_list = []
    #Get the classes
    for folder in glob.glob(folder_name+'/*/*'):
        class_list.append(os.path.basename(folder))
    fields = ['index','class_name']
    values = []
    print('Number of classes:',len(class_list))
    for i,class_name in enumerate(class_list):
        values.append([i, class_name])
    file_path = str(Path(path))
    
    #Write to the csv file
    with open(file_path,'w') as f:
        writer = csv.writer(f,lineterminator='\n')

        writer.writerow(fields)
        writer.writerows(values)
    print('Class mappings created successfully and saved to',file_path)

def check_images(folder_name=None):
    """
    This function is used to check images for any inconsistencies in the exif data

    Attributes
    ----------
    folder_name : str
        The name of the folder which contains the images
    """
    print('\nChecking images\n------------')
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    files_list = glob.glob(str(folder_name)+'/*/*/*.jpg')
    count = 0
    for i,jpg in enumerate(files_list):
        sys.stdout.write("\rProcessing image %i of %i files" % (i,len(files_list)))
        #Successfull if exif data is okay
        try:
            img = Image.open(jpg)
            img.close()
        #If corrupted, use cv2 to read and save it
        except:
            #print(jpg)
            count += 1
            img = cv2.imread(jpg)
            cv2.imwrite(jpg,img)
    sys.stdout.write("\r%i images checked successfully, fixed %i files\n" % (len(files_list),count))

def detect_faces_and_save(mtcnn,folder_name=None):
    """
    This function is used to detect faces in the train images and save them. MTCNN algorithm is used.

    Attributes
    ----------
    mtcnn : MTCNN
        Instance of the MTCNN class
    folder_name : str
        The name of the folder which contains the images
    """
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    folder_name = Path(folder_name)
    print('\nDetecting faces and creating cropped images in trimmed_{0}'.format(folder_name))
    files_list = glob.glob(str(folder_name)+'/*/*/*.jpg')
    start = time.time()
    count = 0
    for i,jpg in enumerate(files_list):
        sys.stdout.write("\rProcessing file %i of %i files" % (i,len(files_list)))
        with Image.open(jpg) as img:
            img = img.convert('RGB')
            #img = np.array(img)
            try:
                #Detect faces and save them
                jpg_path = Path(jpg).absolute()
                jpg_path_parts = list(jpg_path.parts)
                jpg_path_parts[-4] = 'trimmed_' + str(jpg_path_parts[-4])
                jpg_path = Path(*jpg_path_parts)
                mtcnn(img,str(jpg_path))
                count += 1
            #Skip any mtcnn is unable to process
            except Exception as e:
                print(e)
                sys.stdout.write("\rSkipping file %s\n" % (str(Path(jpg))))
                #print(jpg)
                #print(e)
        #break
    end = time.time()
    sys.stdout.write("\rProcessed %i files in %s\n" % (count,timedelta(seconds=(end-start)).__str__()))
    

def augment_images(folder_name=None):
    """
    This function is augment images to increase the size of the dataset, horizontal flip is applied.The augmented images are saved at the same location.

    Attributes
    ----------
    folder_name : str
        The name of the folder which contains the images to augment
    """
    print('\nAugmenting images in {0}\n------------'.format(folder_name))
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    files_list = glob.glob(folder_name+'/*/*/*.jpg')
    for i,image in enumerate(files_list):
        sys.stdout.write("\rProcessing file %i of %i files" % (i,len(files_list)))
        #Open the image
        with Image.open(image) as img:
            #Flip the image
            flipped_img = ImageOps.mirror(img)
            #Create the path where the flipped image would be written
            flipped_img_pth = Path(image)
            flipped_img_pth_parts  = list(flipped_img_pth.parts)
            flipped_img_pth_parts[-1] = flipped_img_pth_parts[-1][:-4]+'_flipped.jpg'
            flipped_img_pth = Path(*flipped_img_pth_parts)
            #saving the flipped image
            flipped_img_pth.parents[0].mkdir(parents=True, exist_ok=True)
            flipped_img.save(flipped_img_pth)
    sys.stdout.write("\r%i images augmented successfully\n" % (len(files_list)))
        

def zip_dataset(folder_name=None,zip_path='processed_dataset.zip'):
    """
    This function is zips the dataset and deletes the temporary folder.

    Attributes
    ----------
    folder_name : str
        The name of the folder which contains the final images
    zip_path : str
        path where to save the zip file
    """
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    folder_name_path = Path(folder_name)
    parent = folder_name_path.parent
    stem = folder_name_path.stem
    folder_name_path = Path(parent).joinpath(stem)
    print('\nSaving dataset in {0}'.format(folder_name_path))
    shutil.make_archive(folder_name_path,'zip',folder_name)
    print('Cleaning up....')
    shutil.rmtree(str(folder_name_path))
    print('Dataset saved in {0}.zip successfully!'.format(folder_name_path))

def create_dataset_from_folder(folder_name=None,image_size=160,device='cpu',zip_path=None):
    """
    This function creates a dataset from a given folder of images

    Attributes
    ----------
    folder_name : str
        The name of the folder which contains the images
    image_size : int
        The size of the cropped image
    device : string
        The device to use,gpu or cpu (default: cpu)
    """
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    print('Creating dataset from folder',folder_name)
    mtcnn = MTCNN(
        image_size=image_size, margin=5, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    #Create class mappings
    create_class_mappings(folder_name)
    #Check all images
    check_images(folder_name)
    #Detect faces and save them
    detect_faces_and_save(mtcnn,folder_name)
    del mtcnn
    folder_name_path = Path(folder_name)
    folder_name_path_parts = list(folder_name_path.parts)
    folder_name_path_parts[-1] = 'trimmed_{0}'.format(str(folder_name_path_parts[-1]))
    folder_name_path = Path(*folder_name_path_parts)
    augment_images(str(folder_name_path))
    if zip_path is not None:
        zip_dataset(folder_name=str(folder_name_path),zip_path=zip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a dataset")
    parser.add_argument('folder' ,type=str,nargs=1, help="folder which will be used to build the dataset")
    parser.add_argument('-s',metavar='--image-size', type=int,nargs='?',default=160,help="the size of the image")
    parser.add_argument('--zip',nargs=1,type=str,help="If specified zips the dataset")
    args = parser.parse_args()
    parsed_args = vars(args)
    zip_path = None
    if parsed_args['zip']:
        zip_path = parsed_args['zip'][0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    folder_name = str(Path(parsed_args['folder'][0]).absolute)
    create_dataset_from_folder(folder_name=folder_name,image_size=parsed_args['s'],device=device,zip_path=zip_path)
