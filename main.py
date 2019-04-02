from __future__ import print_function
from __future__ import division

from backbone_cnn.customize_squeezenet import squeezenet_customize
from torchsummary import summary
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import math
import os
import copy
import shutil
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
import argparse

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

parser = argparse.ArgumentParser(
    description='Customize SqueezeNet for Image Classification With Pytorch')

parser.add_argument("--mode", default="train", type=str,
                    help='Choose train or test mode.')

args = parser.parse_args()

input_size = 224

data_dir = "./data/Car"

num_classes = 2

batch_size = 4

num_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

class_names = image_datasets['train'].classes

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

for x in ['train', 'val', 'test']:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
print(image_datasets['train'].classes)

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)
    
def show_databatch(inputs, classes):
    out=torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    
    model.train(False)
    model.eval()
    
    images_so_far = 0
    
    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        size = inputs.size()[0]
        
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        
        outputs = model(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        # print("Ground truth:")
        # show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far +=size
        if images_so_far >= num_images:
            break
            
    model.train(mode=was_training)

def eval_model(model, criterion, dataloaders, dataset_sizes):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders['test'])
    print("Evaluating model")
    print('-' * 100)
    
    for i, data in enumerate(dataloaders['test']):
        if i % 100 == 0:
            print("\r Test batch {}/{} \n".format(i, test_batches), end='', flush=True)
            
        model.train(False)
        model.eval()
        inputs, labels = data
        
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        
        outputs = model(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        loss_test += loss.item() * inputs.size(0)
        acc_test += torch.sum(preds == labels.data)
        
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes['test']
    avg_acc = acc_test.double() / dataset_sizes['test']
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 100)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './checkpoint/{}.pth'.format('sq_custom_model'))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    squeezenet_release = squeezenet_customize(pretrained=False)

    squeezenet_release.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)) 

    squeezenet_release.num_classes = num_classes

    squeezenet_release= squeezenet_release.to(device)

    if args.mode == 'train':
        summary(squeezenet_release, input_size=(3, 224, 224))
        optimizer = optim.SGD(squeezenet_release.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        trained_model = train_model(squeezenet_release, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    if args.mode == 'test':
        optimizer = optim.SGD(squeezenet_release.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        squeezenet_release.load_state_dict(torch.load('./checkpoint/{}.pth'.format('sq_custom_model'), map_location=device)) # load model from checkpoint
        summary(squeezenet_release, input_size=(3, 224, 224)) # summary model with parameters
        # eval_model(squeezenet_release, criterion, dataloaders_dict, dataset_sizes) # use it if you want to evaluate the model for all test set
        visualize_model(squeezenet_release, dataloaders_dict, num_images=20) #visualize model to compare between ground truth and predicted
