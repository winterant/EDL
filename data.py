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


def load_scene(train_rate=0.8, img_resize=(28, 28)):
    x, y = [], []
    counter = 0
    for label_dir in Path(r'E:\dataset\vision\15-Scene').iterdir():
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


def load_scene_hog(train_rate=0.8):
    # scene15-gist-hog-lbp.pkl is generated using https://gitee.com/winterantzhao/image-feature-extractor
    x, y = pickle.load(open(r'E:\project\ai\uncertainty\datasets\scene15-gist-hog-lbp.pkl', 'rb'))
    rand_indices = np.random.permutation(len(y))
    x = MinMaxScaler([0, 1]).fit_transform(x['hog']).astype(np.float32)  # Normalization
    x = torch.Tensor(x[rand_indices])
    y = torch.Tensor(y[rand_indices].flatten()).long()
    num_train = int(len(y) * train_rate)
    data_train = TensorDataset(x[:num_train], y[:num_train])
    data_val = TensorDataset(x[num_train:], y[num_train:])
    return {
        "classes": 15,
        "train": data_train,
        "val": data_val,
    }


def load_notmnist(train_rate=0.8, img_resize=(28, 28)):
    path = Path(r'E:\project\ai\uncertainty\datasets\notmnist\notMNIST_small.mat')
    notmnist = scipy.io.loadmat(str(path))
    notmnist_x = torch.FloatTensor(notmnist["images"]).permute([2, 0, 1]).reshape([-1, 1] + img_resize) / 255  # Normalization
    notmnist_y = torch.FloatTensor(notmnist["labels"])

    dataset = TensorDataset(notmnist_x, notmnist_y)
    num_train = int(len(dataset) * train_rate)
    data_train, data_val = torch.utils.data.random_split(dataset, [num_train, len(dataset) - num_train])
    return {
        "classes": notmnist_y.max() + 1,  # normally 10.
        "train": data_train,
        "val": data_val,
    }


def load_mnist():
    path = Path(r'E:\project\ai\uncertainty\datasets\mnist')
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    data_train = MNIST(str(path), train=True, download=True, transform=trans)
    data_val = MNIST(str(path), train=False, download=True, transform=trans)
    return {
        "classes": 10,
        "train": data_train,
        "val": data_val,
    }


def load_cifar10():
    path = Path(r'E:\project\ai\uncertainty\datasets\cifar10').__str__()
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    data_train = CIFAR10(path, train=True, download=True, transform=trans)
    data_val = CIFAR10(path, train=False, download=True, transform=trans)
    return {
        "classes": 10,
        "train": data_train,
        "val": data_val,
    }