import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from datetime import datetime
from torch.utils.data import WeightedRandomSampler, Dataset
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
from imgaug import augmenters as iaa
import imgaug as ia

Image.MAX_IMAGE_PIXELS = None


class ClassificationMapedDataset(Dataset):
    def __init__(self, dataset_path, class_to_class_map, transform=None, ignore_list=['']):
        self.ignore_list = ignore_list
        classes, class_to_idx = self._find_classes(dataset_path, class_to_class_map)
        self.classes = classes
        self.class_to_idx = class_to_idx
        samples = self._make_dataset(dataset_path, class_to_idx)
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = datasets.folder.default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def _find_classes(self, dir, class_to_class_map):
        if class_to_class_map is not None:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            # remove ignore classes
            for ignore_class in self.ignore_list:
                try:
                    classes.remove(ignore_class)
                    class_to_idx.pop(ignore_class, None)
                except:
                    pass

            uniq_clases_key_map = []
            uniq_clases_value_map = []
            for key, value in class_to_class_map.items():
                uniq_clases_key_map.append(key)
                uniq_clases_value_map.append(value)
            uniq_clases_key_map = list(set(uniq_clases_key_map))
            uniq_clases_value_map = list(set(uniq_clases_value_map))
            # swap value in self.class_to_idx for values in
            for key, value in class_to_idx.items():
                print(class_to_class_map)
                class_to_idx[key] = class_to_idx[class_to_class_map[key]]
            # classes reduce to map calsses
            classes = uniq_clases_value_map
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dataset_path, class_to_idx):
        images = []
        dir = os.path.expanduser(dataset_path)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        return images


class ImgAugTransform:
    def __init__(self, agumentation = {}):
        try:
            self.aug = iaa.Sequential([
                iaa.Resize((224, 224)),
                iaa.Sometimes(
                    float(agumentation['GaussianBlur_P']),
                    iaa.GaussianBlur(sigma=(agumentation["GaussianBlur_sigma"]))
                ),
                iaa.Sometimes(
                    float(agumentation['MedianBlur_P']),
                    iaa.imgcorruptlike.MotionBlur(severity=agumentation['MedianBlur_Severity'])
                ),
                iaa.Sometimes(
                    agumentation['DefocusBlur_P'],
                    iaa.imgcorruptlike.DefocusBlur(severity=agumentation['DefocusBlur_Severity'])
                ),
                iaa.Sometimes(
                    agumentation['Cutout_P'],
                    iaa.Cutout(nb_iterations=agumentation['Cutout_cum_of_cuts']),
                ),
                iaa.ChannelShuffle(agumentation['ChannelSuffle_P']),
                iaa.Sometimes(
                    agumentation['ElasticTransformation_P'],
                    iaa.ElasticTransformation(alpha=agumentation['ElasticTransformation_alpha'],
                                            sigma = agumentation['ElasticTransformation_sigma']
                    ),
                ),
                iaa.Sometimes(
                    agumentation['PerspectiveTransform_P'],
                    iaa.PerspectiveTransform(scale=agumentation['PerspectiveTransform_scale']),
                ),
            ], random_order=True)
        except:
            self.aug = iaa.Sequential([
                iaa.Resize((224, 224)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                iaa.Dropout2d(p=0.5),
                iaa.Sometimes(0.5,iaa.Cutout(nb_iterations=2)),
                iaa.Sometimes(0.25,iaa.Invert(0.25, per_channel=0.5)),
                iaa.Sometimes(0.25,iaa.Add((-40, 40))),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img).copy()

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        self.callback_get_label = callback_get_label
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class Train:
    def __init__(self, model_type='vgg16', hyperparams={
        "model_type": 'vgg16',
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 10,
        "train_valid_test_split": [0.85, 0.10, 0.05],
    }, folders_structur = {
        "models_folder": "models",
        "graph_folder": "graph_folder",
        "confusion_matrix_folder": "confusion_matrix",
        "test_img_folder": "test_img_folder",
        "metadata_json_folder": "metadata_json"
    }, agumentations = {}, class_map = None, ignore_list =['Object_Detection_Annotations']
    ):
        self.transforms = transforms.Compose([
                ImgAugTransform(),
                lambda x: Image.fromarray(x),
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
        self.class_map = class_map
        self.ignore_list = ignore_list

        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.graph_folder, exist_ok=True)
        os.makedirs(self.test_img_folder, exist_ok=True)
        os.makedirs(self.confusion_matrix_folder, exist_ok=True)
        os.makedirs(self.json_folder, exist_ok=True)

    def train(self, dataset_path):
        dataset = ClassificationMapedDataset(dataset_path=dataset_path,
                                            class_to_class_map=self.class_map,
                                            transform=self.transforms,
                                            ignore_list=self.ignore_list)
        # dataset = datasets.ImageFolder(dataset_path, self.transforms)
        train_loader, valid_loader, test_loader, classes = self._split_dataset_to_dataloaders_and_return_classes(dataset)
        model = self._load_model(classes)
        # criterion and optim
        criterion = nn.CrossEntropyLoss()
        try:
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=self.learning_rate)
        except Exception as e:
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        # start train loop
        scaler = torch.cuda.amp.GradScaler()
        valid_loss_min = np.Inf
        valid_acc_min = 0.0
        graph_train_loss = []
        graph_valid_loss = []
        graph_valid_acc = []
        for epoch in range(self.epochs):
            print("epoch start :", epoch)
            start = time.time()
            train_loss = 0.0
            valid_loss = 0.0
            model.train()
            for inputs, labels in train_loader:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logps = model(inputs)
                loss = criterion(logps, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
        graph_plot_path = os.path.join(self.graph_folder,'plot.jpg')
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
        self.classes = classes
        train_size = int(self.train_valid_test_split[0] * len(dataset))
        valid_size = int(self.train_valid_test_split[1] * len(dataset))
        test_size = int(self.train_valid_test_split[2] * len(dataset))
        rest = len(dataset) - train_size - valid_size - test_size
        train_size =  train_size + rest
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
                                                   num_workers=0, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=0, pin_memory=True)

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
        elif self.model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            # model.classifier[6].out_features = nn.Linear(4096, len(self.classes))
            model.classifier = nn.Sequential(
                        nn.Linear(25088, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, len(classes))).to(self.device)
            model = model.to(self.device)
        return model

    def _generate_output_dict(self, conf_matrix, confusion_matrix_path, graph_plot_path,
                                test_accuracy, test_images, true_label_list, pred_label_list,
                                model_path, graph_train_loss, graph_valid_loss, graph_valid_acc):
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
    print('start')
    kjn = Train()
    output_dict = kjn.train(dataset_path='D:/universal image classification 2/test_dataset')
    # dataset = datasets.ImageFolder('test_dataset')
    # model = kjn._load_model()
    # print(model)
