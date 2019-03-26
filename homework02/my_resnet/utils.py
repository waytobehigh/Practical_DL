import torch
import torchvision
import numpy as np
import pandas as pd

from skimage.io import imread
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from copy import deepcopy
from os import path


class TestLoader(DatasetFolder):
    def __init__(self, root, annotation_file, class_to_label, transform=None):
        self.annotation = pd.read_csv(annotation_file, sep='\t', header=None)
        self.class_to_label = class_to_label
        self.root = root
        self.transform = transform

    def __len__(self):
        return self.annotation.shape[0]

    def __getitem__(self, idx):
        img_path = path.join(self.root, self._get_name(idx))
        image = imread(img_path)

        if image.ndim != 3:
            image = np.concatenate([image[:, :, np.newaxis] for _ in range(3)], axis=-1)

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_label[self._get_class(idx)]

    def _get_class(self, idx):
        return self.annotation.iloc[idx, 1]

    def _get_name(self, idx):
        return self.annotation.iloc[idx, 0]


def make_datasets(dataset_dir):
    augment_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ]

    basic_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    train_transform = transforms.Compose(augment_transforms + basic_transforms)

    dataset = torchvision.datasets.ImageFolder(path.join(dataset_dir, 'train/'), transform=train_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])

    test_transform = transforms.Compose(basic_transforms)

    val_dataset = deepcopy(val_dataset)
    val_dataset.dataset.transform = test_transform

    test_dataset = TestLoader(
        path.join(dataset_dir, 'val/images/'), path.join(dataset_dir, 'val/val_annotations.txt'),
        train_dataset.dataset.class_to_idx, transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def make_datagens(datasets, **dataloader_kwargs):
    return (torch.utils.data.DataLoader(dataset, **dataloader_kwargs) for dataset in datasets)
