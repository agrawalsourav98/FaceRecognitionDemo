import torch
from torch import nn
import argparse
import sys
from pathlib import Path

sys.path.insert(0, 'dataset/')

from mtcnn.mtcnn import MTCNN
from dataset.load_dataset import LoadDatasetFromFolder, CreateTrainValDatasets,FaceDataset
from dataset.create_dataset import detect_faces_and_save
from train import create_batched_loader,val_evaluation
from models.resnet50 import resnet50


if __name__ == "__main__":
    #Parse the arguments
    parser = argparse.ArgumentParser(description="Test on a trained model")
    parser.add_argument('folder' ,type=str,nargs=1, help="folder which will be used to build the dataset")
    parser.add_argument('--weight-file',type=str,nargs=1, help="The weight file for initializing the weights")
    parser.add_argument('--weight-type',type=str,nargs=1, help="The weight file type")
    parser.add_argument('--class-list-file',default='class_list.csv',type=str,nargs=1,help="The csv file containing class embeddings")
    args = parser.parse_args()
    parsed_args = vars(args)
    #Select device touse
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    #Crop faces from the given images
    mtcnn = MTCNN(image_size=160, margin=5, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device)
    detect_faces_and_save(mtcnn,parsed_args['folder'][0])
    folder_name = Path(parsed_args['folder'][0])
    folder_name = Path(folder_name.parent).joinpath('trimmed_'+str(folder_name.name))
    folder_name = str(folder_name)
    #Load the dataset
    dataset = LoadDatasetFromFolder(folder_name,parsed_args['class_list_file'][0])
    class_mappings = dataset.get_class_mappings()
    class_list = dataset.get_classes_list()
    X,y = dataset.load()
    testdataset = FaceDataset(X,y,transform=True)
    print('Test set is of length',testdataset.__len__())
    batch_size = 16
    #Create the loader
    testloader = create_batched_loader(testdataset,batch_size=batch_size)
    model = resnet50(num_classes=dataset.num_classes())
    model.load_state_dict(torch.load(parsed_args['weight_file'][0])['weights'])
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    #Perform the evaluation
    val_evaluation(testloader,model,loss_fn,device,test=True)
