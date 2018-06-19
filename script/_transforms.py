
import random
import numbers
import numpy as np
import torch


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        # >>> transforms.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2=None):
        if img2 is None:
            for t in self.transforms:
                img1 = t(img1)
            return img1
        else:
            for t in self.transforms:
                img1, img2 = t(img1, img2)
            return img1, img2


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img1, img2=None):
        if len(img1.shape) == 2:  # this image is grayscale
            img1 = np.expand_dims(img1, axis=0)
        elif len(img1.shape) == 3:  # image is either RGB or YCbCr colorspace
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            img1 = img1.transpose((2, 0, 1))
        img1 = torch.from_numpy(img1).float()
        if img2 is None:
            return img1
        else:
            if len(img2.shape) == 2:
                img2 = np.expand_dims(img2, axis=0)
            elif len(img2.shape) == 3:
                img2 = img2.transpose((2, 0, 1))
            img2 = torch.from_numpy(img2).float()

            return img1, img2


class CenterCrop(object):
    def __init__(self, output_size):
        if isinstance(output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img1, img2=None):
        h, w = img1.shape[:2]
        new_h, new_w = self.output_size
        top = int(round(h - new_h) * 0.5)
        left = int(round(w - new_w) * 0.5)
        if img2 is None:
            return img1[top: top + new_h, left: left + new_w]
        else:
            return img1[top: top + new_h, left: left + new_w], img2[top: top + new_h, left: left + new_w]


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        if isinstance(output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img1, img2=None):
        h, w = img1.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        if img2 is None:
            return img1[top: top + new_h, left: left + new_w]
        else:
            return img1[top: top + new_h, left: left + new_w], img2[top: top + new_h, left: left + new_w]


class Color0_255to1_1(object):
    """
    [0, 255] to [-1, 1]
    """
    def __call__(self, img1, img2=None):
        if img2 is None:
            return img1/127.5 - 1.
        else:
            return img1/127.5 - 1., img2/127.5 - 1.


def Color1_1to0_255(img):
    """
    [-1, 1] to [0, 255]
    """
    return (img + 1.) * 127.5


def tensor_to_img(input):
    input = torch.squeeze(input)
    input = input.cpu().numpy()
    if input.shape[0] == 3:
        input = input.transpose((1, 2, 0))
    return input


def batch_to_merged_img(input, tile_shape):
    """
    :param input: batch (tensor)
    :param height: 새로 생성될 타일의 세로 개수
    :param width: 새로 생성될 타일의 가로 개수
    :return: 한장의 cv2를 위한 이미지
    """
    input = input.cpu().numpy()
    input = input.transpose((0, 2, 3, 1))
    merged = np.squeeze(merge(input, tile_shape))
    return merged


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    # color
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]  # 나누기 연산 후 몫이 아닌 나머지를 구함
            j = idx // size[1]  # 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    # gray scale
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


