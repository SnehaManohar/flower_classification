from multiprocessing import freeze_support

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim import Adam

from model import Model

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# Hyperparameters.
EPOCHS = 20
NUM_CLASSES = 5
BATCH_SIZE = 100
LR = 0.001
WORKERS = 4

#augmenting the data so as to  artificially expand 
#the size of a training dataset by creating modified 
#versions of images in the dataset. Here this is done 
#to manage the imbalance in the dataset, we need 
#all classes to have almost same number of training examples
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

DATA_PATH_TRAIN = '../data/train/flowers'
DATA_PATH_TEST = '../data/test/flowers'
MODEL_STORE_PATH = '../model/'

train_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=trans)
test_dataset = datasets.ImageFolder(root=DATA_PATH_TEST, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

is_gpu_available = torch.cuda.is_available()

model = Model(NUM_CLASSES)

if is_gpu_available:
    model.cuda()

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
#This is a multiclass classification problem, using
#CrossEntropyLoss will give us probabilities of data 
#point in every class such that, sum of all probabilities 
#in final layer is 1
loss_fn = nn.CrossEntropyLoss()

def checkpoint(epoch):
    torch.save(model.state_dict(), MODEL_STORE_PATH + str(epoch) + ".model.epoch")
    print(epoch, "model saved")

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if is_gpu_available:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)

        test_acc += torch.sum(prediction == labels.data).float()

    test_acc = test_acc / 4242 * 100

    return test_acc


def train(num_epoch):
    best_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if is_gpu_available:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data).float()

        train_acc = train_acc / 4242 * 100
        train_loss = train_loss / 8484

        test_acc = test()

        if test_acc > best_acc:
            checkpoint(epoch)
            best_acc = test_acc

        print("Epoch %s, Training Acc: %s , Training loss: %s , Test Acc: %s" %({epoch + 1}, {train_acc},{train_loss}, {test_acc}))

if __name__ == '__main__':
    freeze_support()
    train(EPOCHS)
