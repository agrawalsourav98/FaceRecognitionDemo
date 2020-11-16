import pickle
import argparse
import os
import copy
import signal

import torch
import torch.utils
from torch import nn,optim

import numpy as np

from dataset.load_dataset import LoadDatasetFromFolder,CreateTrainValDatasets
from models.resnet50 import resnet50

import sys

def load_init_weights(file_name=None,file_type="pth",model=None,optimizer=None,ignore_weights=['fc.bias','fc.weight'],resume=False):
    if model is None:
        raise ValueError("Model cannot be empty")
    assert os.path.exists(file_name), 'folder: {} not found.'.format(file_name)
    if file_type == 'pth':
        saved_model = torch.load(file_name)
        if resume:
            model.load_state_dict(saved_model['weights'])
            if optimizer is not None:
                optimizer.load_state_dict(saved_model['optimizer_state_dict'])
            return model 
        weights = saved_model
    elif file_type == 'pkl':
        with open(file_name, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')
    
    ignore = ignore_weights
    own_state = model.state_dict()
    parameters = model.named_parameters()
    copied_params = []
    for name, param in weights.items():
        if (name in own_state) and (name not in ignore):
            #print(name)
            try:
                if torch.is_tensor(param):
                   own_state[name].copy_(param)
                else:
                    own_state[name].copy_(torch.from_numpy(param))
                copied_params.append(name)
            except Exception as e:
                #print(name)
                print(e)
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.shape))
    for param in parameters:
        if param[0] in copied_params:
            param[1].requires_grad = False
    return model

def create_batched_loader(dataset=None,batch_size=16,shuffle=True):
    if dataset is None:
        raise ValueError("Dataset cannot be empty")
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return loader

def evaluation(dataloader, model, device='cpu'):
    total,correct = 0,0
    for data in dataloader:
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        _,pred = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (pred==labels).sum().item()
    return 100 * correct / total

def train_epoch(model=None,loss_fn=None,trainloader=None,optimiser=None,scheduler=None,device='cpu'):
    if model is None:
        raise AttributeError("Model cannot be empty")
    if loss_fn is None:
        raise AttributeError("loss_fn cannot be empty")
    if trainloader is None:
        raise AttributeError("trainloader cannot be empty")
    if optimiser is None:
        raise AttributeError("optimiser cannot be empty")
    min_loss = float("inf")
    total_loss = 0.0
    count = 0
    pred_count = 0
    for i,data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        labels = labels.long()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

          #print('Min loss %0.2f'% min_loss)
        #print(np.array(total_preds).shape)
        _, preds = torch.max(outputs, 1)
        pred_count += len(preds)
        #print(preds)
        #print(labels)
        for pred,label in zip(preds,labels):
            if pred == label:
                count +=1
        #print(count)
        acc = (count)/pred_count
        sys.stdout.write("\rTrain: Iteration %i/%i | Loss: %0.3f | Acc: %0.2f" % (i+1,len(trainloader),(total_loss/(i+1)),acc*100))
        del inputs,labels,outputs
        torch.cuda.empty_cache()
    if scheduler is not None:
        scheduler.step()
    torch.save({'weights':model.state_dict(),'optimizer_state_dict':optimiser.state_dict()},'saved_model.pth')
    sys.stdout.write('\n');    
    return loss.item()

def train_model(model=None,loss_fn=None,trainloader=None,valloader=None,epochs=1,batch_size=16,optimiser=None,scheduler=None,device='cpu'):
    if model is None:
        raise AttributeError("Model cannot be empty")
    if loss_fn is None:
        raise AttributeError("loss_fn cannot be empty")
    if trainloader is None:
        raise AttributeError("trainloader cannot be empty")
    if optimiser is None:
        raise AttributeError("optimiser cannot be empty")
    if valloader is None:
        raise AttributeError("valloader cannot be empty")
    print("Starting the training\n----------------------")
    loss_epoch_arr = []
    n_iters = np.ceil(len(trainloader))
    #print("Validation Accuracy: %0.2f" % evaluation(valloader,model,device))
    for epoch in range(epochs):
        print('\nEpoch %i/%i\n-------------' % (epoch+1,epochs))
        epoch_loss = train_epoch(model=model,loss_fn=loss_fn,trainloader=trainloader,optimiser=optimiser,scheduler=scheduler,device=device)
        loss_epoch_arr.append(epoch_loss)
        total_loss = 0.0
        count = 0
        pred_count = 0
        for i,data in enumerate(valloader):
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            labels = labels.long()
            outputs = model(inputs)
            batch_loss = loss_fn(outputs,labels)
            total_loss += batch_loss
            _, preds = torch.max(outputs, 1)
            pred_count += len(preds)

            for pred,label in zip(preds,labels):
                if pred == label:
                    count +=1
            
            acc = (count)/pred_count
            sys.stdout.write("\rVal: Iteration %i/%i | Loss: %0.3f | Acc: %0.2f" % (i+1,len(valloader),(total_loss/(i+1)),acc*100))
            del inputs,labels,outputs
            torch.cuda.empty_cache()
    torch.save({'weights':model.state_dict(),'optimizer_state_dict':optimiser.state_dict()},'saved_model_to_resume.pth')
    torch.save(model.state_dict(),'saved_model_final.pth')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset")
    parser.add_argument('folder' ,type=str,nargs=1, help="folder which will be used to build the dataset")
    parser.add_argument('--weight-file',type=str,nargs=1, help="The weight file for initializing the weights")
    parser.add_argument('--weight-type',type=str,nargs=1, help="The weight file type")
    parser.add_argument('--resume',action=argparse._StoreTrueAction,help="Resume training")
    #parser.add_argument('-s',metavar='--image-size', type=int,nargs='?',default=160,help="the size of the image")
    args = parser.parse_args()
    parsed_args = vars(args)
    #print(parsed_args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    loaded_dataset = LoadDatasetFromFolder(parsed_args['folder'][0])
    X,y = loaded_dataset.load()
    datasets = CreateTrainValDatasets(X,y)
    trainset = datasets.get_trainset()
    valset = datasets.get_valset()
    print('Train set is of length',trainset.__len__())
    print('Val set is of length',valset.__len__())
    batch_size = 32
    trainloader = create_batched_loader(trainset,batch_size=batch_size)
    valloader = create_batched_loader(valset,batch_size=batch_size)
    model = resnet50(num_classes=loaded_dataset.num_classes())
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(opt,[5,10])
    try:
        if parsed_args['resume'] is not None:
            resume = True
        else:
            resume = False
        model = load_init_weights(parsed_args['weight_file'][0],parsed_args['weight_type'][0],model,optimizer=opt,resume=resume)
        print("Weights loaded successfully")
    except Exception as e:
        print(e)
        print("Weight file not provided,continuing without pre training")
    model = model.to(device)
    train_model(model,loss_fn,trainloader,valloader,optimiser=opt,scheduler=scheduler,device=device)

    #Doing the testing
    #dataset = LoadDatasetFromFolder(parsed_args['folder'][0])
    #X,y = dataset.load()
    #testdataset = FaceDataset(X,y,transform=True)
    #print('Test set is of length',testdataset.__len__())
    batch_size = 1
    testloader = create_batched_loader(valset,batch_size=batch_size)
    model = resnet50(num_classes=1012)
    model.load_state_dict(torch.load('saved_model.pth')['weights'])
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    #print(model)
    model.eval()
    total_loss = 0.0
    count = 0
    pred_count = 0
    for i,data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        labels = labels.long()
        outputs = model(inputs)
        batch_loss = loss_fn(outputs,labels)
        total_loss += batch_loss
        _, preds = torch.max(outputs, 1)
        pred_count += len(preds)
        print(preds,labels)
        for pred,label in zip(preds,labels):
            if pred == label:
                count +=1
        
        acc = (count)/pred_count
        sys.stdout.write("\rTest: Iteration %i/%i | Loss: %0.3f | Acc: %0.2f" % (i+1,len(testloader),(total_loss/(i+1)),acc*100))
        del inputs,labels,outputs,preds
        torch.cuda.empty_cache()
    

    


