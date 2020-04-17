import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from datetime import datetime
from torch.utils.data import WeightedRandomSampler
import time
import os
from PIL import Image
import random
from matplotlib import pyplot as plt
from imbalanced import ImbalancedDatasetSampler

class Train:
    def __init__(self, models_folder, graph_folder):
        self.transforms = transforms.Compose([
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        self.model_folder = models_folder
        self.batch_size = 64
        self.learning_rate = 0.001
        self.data_dim = 5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = 100
        self.graph_folder =graph_folder

    def train(self, dataset_path):
        train_set = datasets.ImageFolder(dataset_path, self.transforms)
        classes = train_set.classes
        print("classes: ", classes)
        # spliting data
        train_size = int(0.95 * len(train_set))
        val_size = len(train_set) - train_size
        # check if all have good size and if not add one
        if (train_size + val_size == len(train_set)):
            train_set, valid_set = torch.utils.data.random_split(train_set,
                                                            [train_size, val_size])
        else:
            try:
                train_set, valid_set = torch.utils.data.random_split(train_set,
                                                                [train_size, val_size+1])
            except:
                train_set, valid_set = torch.utils.data.random_split(train_set,
                                                                [train_size+1, val_size])
        print("post len(train_set): ", len(train_set))
        print("post len(valid_set): ", len(valid_set))
        print("type(train_set): ", type(train_set))



        train_loader = torch.utils.data.DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set),
                                                batch_size=self.batch_size,
                                                   num_workers=0)
        valid_loader= torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=0)



        # load device and res net 50
        device = self.device
        model = models.resnet50(pretrained=True).to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
                       nn.Linear(2048, 128),
                       nn.ReLU(inplace=True),
                       nn.Linear(128, len(classes))).to(device)
        model.state_dict(torch.load('models/resnet50_16_04_2020_0.6293.pth'))
        model = model.to(self.device)
        # criterion and optim
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=self.learning_rate)
        # start train loop
        valid_loss_min = np.Inf
        valid_acc_min = 0.0
        graph_train_loss = []
        graph_valid_loss = []
        graph_valid_acc = []
        for epoch in range(self.epochs):
            print("epoch start :", epoch)
            start = time.time()
            model.train()
            train_loss = 0.0
            valid_loss = 0.0
            for inputs, labels in train_loader:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            model.eval()
            with torch.no_grad():
                accuracy = 0
                for inputs, labels in valid_loader:
                    # print("inputs: ", inputs.size())
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # calculate average losses
            train_loss = train_loss/len(train_loader)
            valid_loss = valid_loss/len(valid_loader)
            valid_accuracy = accuracy/len(valid_loader)
            graph_train_loss.append(train_loss)
            graph_valid_loss.append(valid_loss)
            graph_valid_acc.append(valid_accuracy)
            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                epoch + 1, train_loss, valid_loss, valid_accuracy))
            if valid_loss_min >= valid_loss:
                print('valid_loss  decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min ,
                valid_loss))
                now = datetime.now()
                data = now.strftime("%d_%m_%Y")
                model_save_name = 'resnet50_' + data + '_' + str(round(valid_accuracy, 4)) + '.pth'
                path =  os.path.join(self.model_folder, model_save_name)
                torch.save(model.state_dict(), path)
                valid_loss_min = valid_loss
            print(f"Time per epoch: {(time.time() - start):.3f} seconds")
        # ploting data
        fig = plt.figure()
        plt.plot(list(range(0, self.epochs)), graph_train_loss, label='train_loss')
        plt.plot(list(range(0, self.epochs)), graph_valid_loss, label='valid_loss')
        plt.plot(list(range(0, self.epochs)), graph_valid_acc, label='valid_accuracy')
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("values")
        fig.savefig(self.graph_folder+'/plot.jpg', dpi=fig.dpi)

if __name__ == '__main__':
    kjn = Train(models_folder='models', graph_folder='graphs')
    kjn.train(dataset_path='train')
