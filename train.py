import pickle
import argparse
import os
import copy
import time
from datetime import timedelta

import torch
import torch.utils
from torch import nn,optim

import numpy as np

from dataset.load_dataset import LoadDatasetFromFolder,CreateTrainValDatasets
from models.resnet50 import resnet50

import sys

def load_init_weights(file_name=None,file_type="pth",model=None,optimizer=None,ignore_weights=['fc.bias','fc.weight'],resume=False):
    """
    Used to load a weights into a model

    Attributes
    -----------
    file_name: 
        The name of the weight file
    file_type:
        The type of the weight file
    model:
        The model to which the weights will be loaded
    optimizer:
        The optimizer to which weights will be loaded
    ignore_weights:
        The weights that won't be loaded from the weight file
    """
    if model is None:
        raise ValueError("Model cannot be empty")
    assert os.path.exists(file_name), 'file: {} not found.'.format(file_name)
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
    else:
        raise ValueError('weight file type must be pkl or pth, given',file_type)
    
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
    """
    Create a dataloader

    Attributes
    ----------
    dataset:
        The dataset for which the dataloader must be created
    batch_size:
        The batch size of the dataloader
    shuffle:
        Whether to shuffle the dataset
    """
    if dataset is None:
        raise ValueError("Dataset cannot be empty")
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return loader

def val_evaluation(dataloader, model, device='cpu'):
    """
    Perform evaluation on the validation dataset

    Attributes
    ----------
    dataloader:
        The validation dataloader
    model:
        The model to run eval on
    device:
        The device to use for inference
    """
    model.eval()
    total_loss = 0.0
    total,correct = 0,0
    for i,data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        labels = labels.long()
        #Perfoming eval
        outputs = model(inputs)
        batch_loss = loss_fn(outputs,labels)
        #Adding loss
        total_loss += batch_loss.cpu().detach().float()
        #Calculating correct predictions for accuracy
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds==labels).sum().item()

        acc = 100 * correct / total
        #Write the information about current iteration
        sys.stdout.write("\rVal: Iteration %i/%i | Loss: %0.3f | Acc: %0.2f" % (i+1,len(dataloader),(total_loss/(i+1)),acc))
        del inputs,labels,outputs
        torch.cuda.empty_cache()
    model.train()

def train_epoch(model=None,loss_fn=None,trainloader=None,optimiser=None,scheduler=None,device='cpu',epoch=0,chkpt_dir=None):
    """
        Used to train a epoch
        
        Attributes
        ----------
        model:
            The model to train
        loss_fn:
            The loss function
        trainloader:
            The batched dataloader for train dataset
        optimiser:
            The optimiser to use
        scheduler:
            The learning rate scheduler to use
        device:
            The device to use for training, cpu or gpu (default cpu)
        epoch:
            The current epoch
        chkpt_dir:
            The directory where to store checkpoint files
        
        Returns
        -------
        (float) -> Epoch loss
        
    """
    if model is None:
        raise AttributeError("Model cannot be empty")
    if loss_fn is None:
        raise AttributeError("loss_fn cannot be empty")
    if trainloader is None:
        raise AttributeError("trainloader cannot be empty")
    if optimiser is None:
        raise AttributeError("optimiser cannot be empty")
    total_loss = 0.0
    total,correct = 0,0
    for i,data in enumerate(trainloader):
        start = time.time()
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        labels = labels.long()
        #Compute and backpropagate loss
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        #Calculating correct predictions for accuracy
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds==labels).sum().item()

        acc = 100 * correct / total

        sys.stdout.write("\rTrain: Iteration %i/%i | Loss: %0.3f | Time Taken: %0.3f | Acc: %0.2f" % (i+1,len(trainloader),(total_loss/(i+1)),(time.time()-start),acc))
        del inputs,labels,outputs
        torch.cuda.empty_cache()
    if scheduler is not None:
        scheduler.step()
    if chkpt_dir is not None:
        #Checkpoint the model
        path = Path(chkpt_dir).joinpath(epoch+'_saved_model.pth')
        torch.save({'weights':model.state_dict(),'optimizer_state_dict':optimiser.state_dict(),'epoch':epoch},str(path))
    sys.stdout.write('\n')
    return loss.item()

def train_model(model=None,loss_fn=None,trainloader=None,valloader=None,epochs=1,batch_size=16,optimiser=None,scheduler=None,device='cpu',chkpt=False):
    """
        Used to train the specified models
        
        Attributes
        ----------
        model:
            The model to train
        loss_fn:
            The loss function
        trainloader:
            The batched dataloader for train dataset
        valloader:
            The batched dataloader for validation dataset
        epochs:
            The number of epochs to run (default 1)
        batch_size:
            The batch size to use during training (default 16)
        optimiser:
            The optimiser to use
        scheduler:
            The learning rate scheduler to use
        device:
            The device to use for training, cpu or gpu (default cpu)
        
    """
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
    start = time.time()
    loss_epoch_arr = []
    n_iters = np.ceil(len(trainloader))
    #Perform an inital val evaluation
    val_evaluation(valloader,model,device=device)
    training_folder = None
    if chkpt:
        training_folder = 'training'+str(time.time())
        Path(training_folder).mkdir(parents=True,exists_ok=True)
    # The training loop
    for epoch in range(epochs):
        print('\nEpoch %i/%i\n-------------' % (epoch+1,epochs))
        epoch_loss = train_epoch(model=model,loss_fn=loss_fn,trainloader=trainloader,optimiser=optimiser,scheduler=scheduler,device=device,epoch=epoch,chkpt_dir=training_folder)
        loss_epoch_arr.append(epoch_loss)
        val_evaluation(valloader,model,device=device)
    #Save the weights such that training can be resumed
    torch.save({'weights':model.state_dict(),'optimizer_state_dict':optimiser.state_dict()},'saved_model_to_resume.pth')
    #Save the model for eval
    torch.save(model.state_dict(),'saved_model_final.pth')
    print('\nTraining finished in',timedelta(seconds=time.time()-start))



if __name__ == "__main__":
    #Parse the arguments
    parser = argparse.ArgumentParser(description="Create a dataset")
    parser.add_argument('folder' ,type=str,nargs=1, help="folder which will be used to build the dataset")
    parser.add_argument('--weight-file',type=str,nargs=1, help="The weight file for initializing the weights")
    parser.add_argument('--weight-type',type=str,nargs=1, help="The weight file type")
    parser.add_argument('--resume',action=argparse._StoreTrueAction,help="Resume training")
    parser.add_argument('--class-list-file',type=str,nargs=1,help="The csv file containing class embeddings")
    parser.add_argument('--val-size',type=float,nargs=1,help="The size of the val set")
    parser.add_argument('--stratify',type=argparse._StoreTrueAction,help="Whether to use stratify during train test split")
    parser.add_argument('--shuffle',type=argparse._StoreTrueAction,help="Whether to shuffle the datasets")
    parser.add_argument('--batch-size',default=16,type=int,nargs=1,help="The batch size for the loader")
    parser.add_argument('--lr',default=0.001,type=float,nargs=1,help="The learning rate")
    parser.add_argument('--epochs',default=5,type=int,nargs=1,help="The number of epochs to run the training for")
    parser.add_argument('--checkpoint',type=argparse._StoreTrueAction,help="Specify whether to checkpoint")
    parser.add_argument('--ignore-weights',type=str,nargs='+',default=['fc.bias','fc.weight'],help="The weights to ignore while loading the weight file")
    args = parser.parse_args()
    parsed_args = vars(args)
    #print(parsed_args)

    #Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)
    
    #Load the dataset from the specified folder
    loaded_dataset = LoadDatasetFromFolder(parsed_args['folder'][0],parsed_args['class_list_file'][0])
    X,y = loaded_dataset.load()
    #Create the train-val split
    val_size = 0.12
    if parsed_args['val_size']:
        val_size = parsed_args['val_size'][0]
    datasets = CreateTrainValDatasets(X,y,val_size=val_size,stratify=parsed_args['stratify'])
    trainset = datasets.get_trainset()
    valset = datasets.get_valset()
    print('Train set is of length',trainset.__len__())
    print('Val set is of length',valset.__len__())
    batch_size = parsed_args["batch_size"]
    shuffle = parsed_args['shuffle']
    #Create the batched loaders
    trainloader = create_batched_loader(trainset,batch_size=batch_size,shuffle=shuffle)
    valloader = create_batched_loader(valset,batch_size=batch_size,shuffle=shuffle)
    #Define the model
    model = resnet50(num_classes=loaded_dataset.num_classes())
    #Define the loss_fn
    loss_fn = nn.CrossEntropyLoss()
    #Define the optimizer
    lr = parsed_args['lr']
    opt = optim.Adam(model.parameters(),lr=0.001)
    #Define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(opt,[5,10])
    resume =False
    if parsed_args['resume'] is not None:
            resume = True
    # Try weight loading, if weight file provided.
    try:
        model = load_init_weights(parsed_args['weight_file'][0],parsed_args['weight_type'][0],model,optimizer=opt,resume=resume,ignore_weights=parsed_args['ignore_weights'])
        print("Weights loaded successfully")
    except Exception as e:
        print(e)
        print("Weight file not provided,continuing without pre training")
    #Move model to specified device
    model = model.to(device)
    epochs = parsed_args['epochs'][0]
    #Start the training
    train_model(model,loss_fn,trainloader,valloader,epochs=epochs,batch_size=batch_size,optimiser=opt,scheduler=scheduler,device=device,chkpt=parsed_args['checkpoint'])
    

    


