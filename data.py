import pickle
from pathlib import Path
import numpy as np
import scipy
from scipy.io import loadmat
import cv2
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms


def load_handwritten_concat(path='dataset/handwritten_6views.pkl', train_rate=0.8):
    # You can get "handwritten_6views.pkl" from https://github.com/winterant/TMC/tree/master/dataset
    x, y = pickle.load(open(path, 'rb'))
    x = np.concatenate(list(x.values()), axis=-1)  # Concatenate all views as a vector
    x = MinMaxScaler([0, 1]).fit_transform(x).astype(np.float32)

    x = torch.Tensor(x)
    y = torch.Tensor(y).long()
    dataset = TensorDataset(x, y)
    num_train = int(len(dataset) * train_rate)
    data_train, data_val = torch.utils.data.random_split(dataset, [num_train, len(dataset) - num_train])
    return {
        "classes": 10,
        "train": data_train,
        "val": data_val,
    }


def load_scene(path=r'E:\dataset\vision\15-Scene', train_rate=0.8, img_resize=(28, 28)):
    # Dataset "scene 15" is downloaded from https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177
    x, y = [], []
    counter = 0
    for label_dir in Path(path).iterdir():
        label = label_dir.name
        for img_path in label_dir.iterdir():
            counter += 1
            print('\rloading scene image', counter, end='')
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=img_resize, interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=0)  # add channel
            x.append(img)
            y.append(int(label))
    x = np.array(x)
    x = (x - x.min()) / (x.max() - x.min())  # Normalization
    y = np.array(y)
    np.random.seed(29)
    rand_id = np.random.permutation(len(y))
    x = torch.Tensor(x[rand_id])
    y = torch.Tensor(y[rand_id]).long()
    num_train = int(len(y) * train_rate)
    data_train = TensorDataset(x[:num_train], y[:num_train])
    data_val = TensorDataset(x[num_train:], y[num_train:])
    return {
        "classes": 15,
        "train": data_train,
        "val": data_val,
    }


def load_notmnist(path=r'E:\project\ai\uncertainty\datasets\notmnist\notMNIST_small.mat', train_rate=0.8, img_resize=(28, 28)):
    notmnist = scipy.io.loadmat(str(path))
    notmnist_x = torch.FloatTensor(notmnist["images"]).permute([2, 0, 1]).reshape([-1, 1] + img_resize)
    notmnist_x = (notmnist_x - notmnist_x.min()) / (notmnist_x.max() - notmnist_x.min())  # Normalization
    notmnist_y = torch.FloatTensor(notmnist["labels"])

    dataset = TensorDataset(notmnist_x, notmnist_y)
    num_train = int(len(dataset) * train_rate)
    data_train, data_val = torch.utils.data.random_split(dataset, [num_train, len(dataset) - num_train])
    return {
        "classes": notmnist_y.max() + 1,  # normally 10.
        "train": data_train,
        "val": data_val,
    }


def load_mnist(path=r'E:\project\ai\uncertainty\datasets\mnist', img_resize=(28, 28)):
    trans = transforms.Compose([transforms.Resize(img_resize), transforms.ToTensor()])
    data_train = MNIST(str(path), train=True, download=True, transform=trans)
    data_val = MNIST(str(path), train=False, download=True, transform=trans)
    return {
        "classes": 10,
        "train": data_train,
        "val": data_val,
    }


def load_cifar10(path=r'E:\project\ai\uncertainty\datasets\cifar10', img_resize=(28, 28)):
    trans = transforms.Compose([transforms.Resize(img_resize), transforms.ToTensor()])
    data_train = CIFAR10(path, train=True, download=True, transform=trans)
    data_val = CIFAR10(path, train=False, download=True, transform=trans)
    return {
        "classes": 10,
        "train": data_train,
        "val": data_val,
    }
