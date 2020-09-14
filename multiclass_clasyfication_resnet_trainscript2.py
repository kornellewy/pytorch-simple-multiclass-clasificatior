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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
import random
from matplotlib import pyplot as plt
import torch.utils.data
from sklearn.metrics import confusion_matrix
import itertools
import json

from imbalanced_dataset_sampler import ImbalancedDatasetSampler

class Train:
    def __init__(self, model_type='resnet50', hyperparams={
        "model_type": 'resnet50',
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 1,
        "train_valid_test_split": [0.70, 0.20, 0.10],
    }, folders_structur = {
        "models_folder": "models",
        "graph_folder": "graph_folder",
        "confusion_matrix_folder": "confusion_matrix",
        "test_img_folder": "test_img_folder",
        "metadata_json_folder": "metadata_json"
    }):
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
        self.data_dim = 5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = hyperparams['model_type']
        self.batch_size = hyperparams['batch_size']
        self.learning_rate = hyperparams['learning_rate']
        self.epochs = hyperparams['epochs']
        self.train_valid_test_split = hyperparams['train_valid_test_split']
        self.model_folder = folders_structur["models_folder"]
        self.graph_folder = folders_structur["graph_folder"]
        self.test_img_folder = folders_structur["test_img_folder"]
        self.confusion_matrix_folder = folders_structur["confusion_matrix_folder"]
        self.json_folder = folders_structur["metadata_json_folder"]

    def train(self, dataset_path):
        dataset = datasets.ImageFolder(dataset_path, self.transforms)
        train_loader, valid_loader, test_loader, classes = self._split_dataset_to_dataloaders_and_return_classes(dataset)
        model = self._load_model(classes)
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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
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
                model_save_name = 'resnet50_' + data + '_valid_acc_' + str(round(valid_accuracy, 4)) + '.pth'
                path =  os.path.join(self.model_folder, model_save_name)
                torch.save(model.state_dict(), path)
                valid_loss_min = valid_loss
            print(f"Time per epoch: {(time.time() - start):.3f} seconds")
        model.eval()
        test_loss = 0.0
        test_accuracy = []
        test_images = []
        true_label_list = []
        pred_label_list = []
        with torch.no_grad():
            for inputs, true_label in test_loader:
                test_images.append(inputs)
                true_label = true_label.tolist()[0]
                true_label_list.append(true_label)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = model(inputs)
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                top_class = top_class.tolist()[0]
                top_class = top_class.pop()
                pred_label_list.append(top_class)
                equals = top_class == true_label
                test_accuracy.append(equals)
        test_accuracy = round(np.mean(test_accuracy), 4)
        print(test_accuracy)
        now = datetime.now()
        data = now.strftime("%d_%m_%Y")
        model_save_name = 'resnet50_final_' + data + '_test_acc_' + str(round(test_accuracy, 4)) + '.pth'
        path =  os.path.join(self.model_folder, model_save_name)
        torch.save(model.state_dict(), path)
        test_images = self._save_and_convert_test_images_to_paths(test_images)
        # ploting data
        graph_plot_path = os.path.join(self.graph_folder+'plot.jpg')
        fig = plt.figure()
        plt.plot(list(range(0, self.epochs)), graph_train_loss, label='train_loss')
        plt.plot(list(range(0, self.epochs)), graph_valid_loss, label='valid_loss')
        plt.plot(list(range(0, self.epochs)), graph_valid_acc, label='valid_accuracy')
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("values")
        fig.savefig(graph_plot_path, dpi=fig.dpi)
        # confusion matrix 
        confusion_matrix_path = os.path.join(self.confusion_matrix_folder, 'plot.jpg')
        conf_matrix = confusion_matrix(true_label_list, pred_label_list)
        plot_confusion_matrix(cm=conf_matrix, target_names=[x for x in range(len(classes))],
                                save_path=confusion_matrix_path)
        conf_matrix = conf_matrix.tolist()
        output_dict = self._generate_output_dict(conf_matrix, confusion_matrix_path, graph_plot_path, test_accuracy, test_images, true_label_list, pred_label_list, path,
                                                 graph_train_loss, graph_valid_loss, graph_valid_acc)
        json_path = os.path.join(self.json_folder, 'metadata.json')
        self._dump_dict_to_json(output_dict, json_path)
        return output_dict 

    def _split_dataset_to_dataloaders_and_return_classes(self, dataset):
        classes = dataset.classes
        train_size = int(self.train_valid_test_split[0] * len(dataset))
        valid_size = int(self.train_valid_test_split[1] * len(dataset))
        test_size = int(self.train_valid_test_split[2] * len(dataset))
        if (train_size + valid_size + test_size == len(dataset)):
            train_set, valid_set, test_set = torch.utils.data.random_split(dataset,
                                                        [train_size, valid_size, test_size])
        else:
            try:
                train_set, valid_set, test_set = torch.utils.data.random_split(dataset,
                                                        [train_size+1, valid_size, test_size])
            except:
                train_set, valid_set, test_set = torch.utils.data.random_split(dataset,
                                                        [train_size, valid_size+1, test_size])                     
        train_loader = torch.utils.data.DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set),
                                                batch_size=self.batch_size,
                                                   num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=0)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
        return train_loader, valid_loader, test_loader, classes

    def _load_model(self, classes, model_path = None):
        if model_path is None:
            model = self._load_specific_model(classes)
        else:
            # TODO load model from path
            model = self._load_specific_model(classes)
        return model

    def _load_specific_model(self, classes):
        if self.model_type == 'resnet50':
            model = models.resnet50(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, len(classes))).to(self.device)
            model = model.to(self.device)
        if self.model_type == 'resnet18':
            model = models.resnet18(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, len(classes))).to(self.device)
            model = model.to(self.device)
        elif self.model_type == 'resnet34':
            model = models.resnet34(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, len(classes))).to(self.device)
            model = model.to(self.device)
        elif self.model_type == 'resnet101':
            model = models.resnet101(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, len(classes))).to(self.device)
            model = model.to(self.device)
        elif self.model_type == 'resnet152':
            model = models.resnet152(pretrained=True).to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, len(classes))).to(self.device)
            model = model.to(self.device)
        return model

    def _generate_output_dict(self, conf_matrix, confusion_matrix_path, graph_plot_path,test_accuracy, test_images, true_label_list, pred_label_list, model_path, graph_train_loss, graph_valid_loss, graph_valid_acc):
        """
        TODO
        """
        output_dict = {
            'conf_matrix': conf_matrix,
            "confusion_matrix_path": confusion_matrix_path,
            "graph_plot_path": graph_plot_path,
            "test_accuracy": test_accuracy,
            "test_images": test_images,
            "true_label_list": true_label_list,
            "pred_label_list": pred_label_list,
            "model_path": model_path, 
            "epochs": [x for x in range(len(graph_train_loss))],
            "train_loss": graph_train_loss,
            "valid_loss": graph_valid_loss,
            "valid_acc": graph_valid_acc,
        }
        return output_dict

    def _save_and_convert_test_images_to_paths(self, test_images):
        test_images_paths = []
        for idx, test_image in enumerate(test_images):
            test_image = self._inv_normalize_tensor(test_image)
            img_path = os.path.join(self.test_img_folder, str(idx)+'.jpg')
            torchvision.utils.save_image(test_image, img_path)
            test_images_paths.append(img_path)
        return test_images_paths

    def _inv_normalize_tensor(self, pytorch_tensor):
        pytorch_tensor = torch.squeeze(pytorch_tensor, 0)
        inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
        )
        inv_pytorch_tensor = inv_normalize(pytorch_tensor)
        return inv_pytorch_tensor

    def _dump_dict_to_json(self, dict_to_dump, json_path):
        with open(json_path, 'w') as fp:
            json.dump(dict_to_dump, fp,  indent=4)


def plot_confusion_matrix(cm, save_path,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fig.savefig(save_path, dpi=fig.dpi)
    return fig

if __name__ == '__main__':
    kjn = Train()
    output_dict = kjn.train(dataset_path='test_dataset')
    import pprint
    pprint.pprint(output_dict)
    # kjn._save_and_convert_test_images_to_paths(output_dict["test_images"])

    # 
    # dataset = datasets.ImageFolder('test_dataset')
    # model = kjn._load_model()
    # print(model)
