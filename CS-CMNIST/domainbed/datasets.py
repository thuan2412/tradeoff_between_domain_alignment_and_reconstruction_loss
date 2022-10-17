import os
from collections import defaultdict

import PIL
from PIL import Image, ImageFile
# from domainbed.lib.corrupted_cifar10_protocol import CORRUPTED_CIFAR10_PROTOCOL
import h5py

import numpy as np

import torch
from utils import TensorDataset
from utils import TensorDataset
from utils import TensorDataset, save_image
from torchvision import transforms
import torchvision.datasets.folder
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from tqdm import tqdm
import io
import pdb
import functools

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "Debug28",
    "Debug224",
    "FullColoredMNIST",
    "ColoredMNIST",
    "RotatedMNIST",
    "VLCS",
    "PACS",
]

NUM_ENVIRONMENTS = {
    "Debug28": 3,
    "Debug224": 3,
    "FullColoredMNIST": 3,
    "ColoredMNIST": 3,
    "RotatedMNIST": 6,
    "VLCS": 4,
    "PACS": 4,
}


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8


class Debug(MultipleDomainDataset):
    DATASET_SIZE = 16
    INPUT_SHAPE = None  # Subclasses should override

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.environments = [0, 1, 2]
        self.datasets = []
        for _ in range(len(self.environments)):
            self.datasets.append(
                TensorDataset(
                    torch.randn(self.DATASET_SIZE, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (self.DATASET_SIZE,))
                )
            )

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENT_NAMES = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENT_NAMES = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        root = os.path.join(root, "MNIST/")

        self.colors = torch.FloatTensor(
            [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225],
             [0, 0, 255], [255, 20, 147], [160, 160, 160]])
        self.random_colors = torch.randint(255, (10, 3)).float()
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        img_shuffle = torch.randperm(len(original_images))
        original_images = original_images[img_shuffle]
        original_labels = original_labels[img_shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(self.environments)):
            images = original_images[i::len(self.environments)]
            labels = original_labels[i::len(self.environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class FullColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENT_NAMES = [0, 1, 2]

    def __init__(self, root, test_envs, hparams):
        self.data_type = hparams['data_type']
        if self.data_type == 0:
            self.ratio = hparams.get('ratio', 0.9)
            self.env_seed = hparams.get('env_seed', 1)
            MY_COMBINE = [[self.env_seed, True, 0.0], [self.env_seed, True, 1.0], [self.env_seed, True, self.ratio]]
        else:
            raise NotImplementedError

        # print("MY COMBINE:", MY_COMBINE)
        super(FullColoredMNIST, self).__init__(root, MY_COMBINE, self.color_dataset, (3, 28, 28,), 10)
        self.input_shape = (3, 28, 28)
        self.num_classes = 10

    def color_dataset(self, images, labels, environment):
        # set the seed
        original_seed = torch.initial_seed()  
        torch.manual_seed(environment[0])
        shuffle = torch.randperm(len(self.colors))
        self.colors_ = self.colors[shuffle] if environment[1] else self.random_colors[shuffle]
        torch.manual_seed(environment[0])
        ber = self.torch_bernoulli_(environment[2], len(labels))
        # print("ber:", len(ber), sum(ber))
        torch.manual_seed(original_seed)  

        images = torch.stack([images, images, images], dim=1)
        # binarize the images
        images = (images > 0).float()
        y = labels.view(-1).long()
        color_label = torch.zeros_like(y).long()

        # Apply the color to the image
        for img_idx in range(len(images)):
            if ber[img_idx] > 0:
                color_label[img_idx] = labels[img_idx]  
                for channels in range(3):
                    images[img_idx, channels, :, :] = images[img_idx, channels, :, :] * \
                                                      self.colors_[labels[img_idx].long(), channels]
            else:
                color = torch.randint(10, [1])[0]  # random color, regardless of label
                color_label[img_idx] = color
                for channels in range(3):
                    images[img_idx, channels, :, :] = images[img_idx, channels, :, :] * self.colors_[color, channels]

        x = images.float().div_(255.0)

        return TensorDataset(True, x, y, color_label)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()