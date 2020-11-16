import torch
from torch import nn
import argparse
import sys
from dataset.load_dataset import LoadDatasetFromFolder,CreateTrainValDatasets,FaceDataset
from train import create_batched_loader
from models.resnet50 import resnet50


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test on a trained model")
    parser.add_argument('folder' ,type=str,nargs=1, help="folder which will be used to build the dataset")
    parser.add_argument('--weight-file',type=str,nargs=1, help="The weight file for initializing the weights")
    args = parser.parse_args()
    parsed_args = vars(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device',device)

    dataset = LoadDatasetFromFolder(parsed_args['folder'][0])
    X,y = dataset.load()
    testdataset = FaceDataset(X,y,transform=True)
    print('Test set is of length',testdataset.__len__())
    batch_size = 1
    testloader = create_batched_loader(testdataset,batch_size=batch_size)
    model = resnet50(num_classes=1012)
    model.load_state_dict(torch.load(parsed_args['weight_file'][0])['weights'])
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
