from os import listdir
from os.path import join

import torch.utils.data as data
from _dataIO import load_img

from _transforms import *


class PairedImageDataSet(data.Dataset):
    def __init__(self, origin_dir, artifact_dir, transform=None):
        self.artifact_filenames = [join(artifact_dir, x) for x in listdir(artifact_dir)]
        self.origin_filenames = [join(origin_dir, x) for x in listdir(origin_dir)]

        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.artifact_filenames[index])
        target = load_img(self.origin_filenames[index])
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.origin_filenames)


class ImageDataSet(data.Dataset):
    def __init__(self, origin_dir, transform=None):
        self.origin_filenames = [join(origin_dir, x) for x in listdir(origin_dir)]
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.origin_filenames[index])
        if self.transform:
            input = self.transform(input)

        return input

    def __len__(self):
        return len(self.origin_filenames)


def folder_to_batch(folder_dir, transform=None):
    """
    :param folder_dir: batch 로 묶을 이미지들이 들어있는 폴더의 경로
    :param transform: RandomCrop, Color0_255to1_1, ToTensor 등
    :return: 폴더 안의 이미지들을 모두 하나의 batch(자료형 tensor) 로 묶은 것.
    """
    batch_paths = [join(folder_dir, x) for x in listdir(folder_dir)]
    if transform is None:
        batch = [load_img(path) for path in batch_paths]
    else:
        batch = [transform(load_img(path)) for path in batch_paths]
    batch = torch.stack(batch)
    return batch

