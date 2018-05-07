import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import sys

def train(network, train_loader, test_loader, epochs, learning_rate, ravel_init=False):        
    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    train_loss_epochs = []
    test_loss_epochs = []
    train_accuracy_epochs = []
    test_accuracy_epochs = []
    try:
        for epoch in range(epochs):
            losses = []
            accuracies = []
            network.train()
            for X, y in train_loader:
                X = Variable(X)
                if ravel_init:
                    X = X.view(X.size(0), -1)
                y = Variable(y)
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                network.zero_grad()
                prediction = network(X)
                loss_batch = loss(prediction, y)
                loss_batch.backward()
                optimizer.step()
                
                losses.append(loss_batch.data[0])
                if torch.cuda.is_available():
                    accuracies.append((np.argmax(prediction.data.cpu().numpy(), 1)==y.data.cpu().numpy()).mean())
                else:
                    accuracies.append((np.argmax(prediction.data.numpy(), 1)==y.data.numpy()).mean())
            train_loss_epochs.append(np.mean(losses))
            train_accuracy_epochs.append(np.mean(accuracies))
            losses = []
            accuracies = []  
            network.eval()
            for X, y in test_loader:
                X = Variable(X)
                if ravel_init:
                    X = X.view(X.size(0), -1)
                y = Variable(y)
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                    
                prediction = network(X)
                loss_batch = loss(prediction, y)
                
                losses.append(loss_batch.data[0])
                if torch.cuda.is_available():
                    accuracies.append((np.argmax(prediction.data.cpu().numpy(), 1)==y.data.cpu().numpy()).mean())
                else:
                    accuracies.append((np.argmax(prediction.data.numpy(), 1)==y.data.numpy()).mean())
            test_loss_epochs.append(np.mean(losses))
            test_accuracy_epochs.append(np.mean(accuracies))
            sys.stdout.write('\rEpoch {0}... (Train/Test) NLL: {1:.3f}/{2:.3f}\tAccuracy: {3:.3f}/{4:.3f}'.format(
                        epoch, train_loss_epochs[-1], test_loss_epochs[-1],
                        train_accuracy_epochs[-1], test_accuracy_epochs[-1]))
    except KeyboardInterrupt:
        pass
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_epochs, label='Train')
    plt.plot(test_loss_epochs, label='Test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_epochs, label='Train accuracy')
    plt.plot(test_accuracy_epochs, label='Test accuracy')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.show()