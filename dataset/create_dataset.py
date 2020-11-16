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

def create_class_mappings(folder_name=None,path=None):
    print('\nCreating class mappings and saving...')
    assert os.path.exists(folder_name), 'folder: {} not found.'.format(folder_name)
    class_list = []
    for folder in glob.glob(folder_name+'/*/*'):
        class_list.append(os.path.basename(folder))
    fields = ['index','class_name']
    values = []
    print('Number of classes:',len(class_list))
    for i,class_name in enumerate(class_list):
        values.append([i, class_name])
    file_path = 'class_list.csv'
    if path:
        file_path = str(Path(path))+'/class_list.csv'
    
    with open(file_path,'w') as f:
        writer = csv.writer(f)

        writer.writerow(fields)
        writer.writerows(values)
    print('Class mappings created successfully and saved to',file_path)

def check_images(folder_name=None):
    print('\nChecking images\n------------')
    assert os.path.exists(folder_name), 'folder: {} not found.'.format(folder_name)
    files_list = glob.glob(str(folder_name)+'/*/*/*.jpg')
    count = 0
    for i,jpg in enumerate(files_list):
        sys.stdout.write("\rProcessing image %i of %i files" % (i,len(files_list)))
        try:
            img = Image.open(jpg)
            img.close()
        except:
            #print(jpg)
            count += 1
            img = cv2.imread(jpg)
            cv2.imwrite(jpg,img)
    sys.stdout.write("\r%i images checked successfully, fixed %i files\n" % (len(files_list),count))

def detect_faces_and_save(mtcnn,folder_name=None):
    
    assert os.path.exists(folder_name), 'folder: {} not found.'.format(folder_name)
    folder_name = Path(folder_name)
    print('\nDetecting faces and creating cropped images in trimmed_{0}'.format(folder_name))
    files_list = glob.glob(str(folder_name)+'/*/*/*.jpg')
    start = time.time()
    count = 0
    for i,jpg in enumerate(files_list):
        #sys.stdout.write("\x1b[1k")
        sys.stdout.write("\rProcessing file %i of %i files" % (i,len(files_list)))
        with Image.open(jpg) as img:
            img = img.convert('RGB')
            #img = np.array(img)
            try:
                mtcnn(img,'trimmed_'+str(jpg))
                count += 1
            except Exception as e:
                sys.stdout.write("\rSkipping file %s\n" % (str(Path(jpg))))
                #print(jpg)
                #print(e)
        #break
    end = time.time()
    sys.stdout.write("\rProcessed %i files in %s\n" % (count,timedelta(seconds=(end-start)).__str__()))
    

def augment_images(folder_name=None):
    print('\nAugmenting images in {0}\n------------'.format(folder_name))
    assert os.path.exists(folder_name), 'folder: {} not found.'.format(folder_name)
    files_list = glob.glob(folder_name+'/*/*/*.jpg')
    for i,image in enumerate(files_list):
        sys.stdout.write("\rProcessing file %i of %i files" % (i,len(files_list)))
        #img = cv2.imread(image)
        #print(image)
        with Image.open(image) as img:
            flipped_img = ImageOps.mirror(img)
            #img_pth = Path(image)
            flipped_img_pth = Path(image)
            #img_pth_parts = list(img_pth.parts)
            flipped_img_pth_parts  = list(flipped_img_pth.parts)
            #saving the flipped image
            flipped_img_pth_parts[-1] = flipped_img_pth_parts[-1][:-4]+'_flipped.jpg'
            #flipped_img_pth_parts[-4] = 'dataset'
            flipped_img_pth = Path(*flipped_img_pth_parts)
            flipped_img_pth.parents[0].mkdir(parents=True, exist_ok=True)
            flipped_img.save(flipped_img_pth)
            #break
            #print(str(flipped_img_pth))
            #cv2.imwrite(str(flipped_img_pth),flipped_img)
            #saving the original image
            #img_pth_parts[-4] = 'dataset'
            #img_pth = Path(*img_pth_parts)
            #img_pth.parents[0].mkdir(parents=True, exist_ok=True)
            #print(str(img_pth))
            #cv2.imwrite(str(img_pth),img)
            #break
    sys.stdout.write("\r%i images augmented successfully\n" % (len(files_list)))
        

def save_dataset(folder_name=None,zip_name='processed_dataset'):
    assert os.path.exists(folder_name), 'folder: {} not found.'.format(folder_name)
    folder_name_path = Path(folder_name)
    print('\nSaving dataset in {0}.zip'.format(folder_name_path))
    shutil.make_archive(folder_name_path.name,'zip',folder_name)
    print('Cleaning up....')
    shutil.rmtree(str(folder_name_path))
    print('Dataset saved in {0}.zip successfully!'.format(folder_name_path))

def create_dataset_from_folder(folder_name=None,image_size=160,device='cpu'):
    assert os.path.isdir(folder_name), 'folder: {} not found.'.format(folder_name)
    print('Creating dataset from folder',folder_name)
    mtcnn = MTCNN(
        image_size=image_size, margin=5, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    check_images(folder_name)
    detect_faces_and_save(mtcnn,folder_name)
    del mtcnn
    folder_name_path = Path(folder_name)
    folder_name_path_parts = list(folder_name_path.parts)
    folder_name_path_parts[-1] = 'trimmed_{0}'.format(str(folder_name_path_parts[-1]))
    folder_name_path = Path(*folder_name_path_parts)
    augment_images(str(folder_name_path))
    save_dataset(folder_name=str(folder_name_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a dataset")
    parser.add_argument('folder' ,type=str,nargs=1, help="folder which will be used to build the dataset")
    parser.add_argument('-s',metavar='--image-size', type=int,nargs='?',default=160,help="the size of the image")
    args = parser.parse_args()
    parsed_args = vars(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    create_dataset_from_folder(folder_name=parsed_args['folder'][0],image_size=parsed_args['s'],device=device)
