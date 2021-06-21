# modified from https://github.com/Lextal/pspnet-pytorch
import random
import numbers
import math
import collections
import torchvision
import statistics 
from scipy.special import softmax
from torchvision import transforms
import torchvision.transforms.functional as F
from collections import Counter
from itertools import groupby

from PIL import ImageOps, Image
import numpy as np
import pickle as cp
import os.path as osp

class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, img):
        return ImageOps.expand(img, border=self.pad, fill=0)


class Scale:
    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        # assert len(imgmap) > 1 # list of images, last one is target (for segmentation tasks only)
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class CenterCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        # imgmap = [i.resize((int(w*1.6),int(h*1.6))) for i in imgmap]
        # w, h = imgmap[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))


        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomSizedCrop:
    def __init__(self, size, interpolation=Image.BICUBIC, consistent=True, p=1.0, clip_len=0, h_ratio=0.7):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p 
        self.clip_len = clip_len
        self.h_ratio = h_ratio

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold: # do RandomSizedCrop
            for attempt in range(10):
                ori_w,ori_h = img1.size
                aspect_ratio = random.uniform(3. / 4, 4. / 3)
                h =  int(random.uniform(self.h_ratio, 1.0) * ori_h)
                w =  int(h*aspect_ratio)
                if self.consistent:
                    # if random.random() < 0.5:
                    #     w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        mid_x = int(img1.size[0]//2)
                        mid_h = int(img1.size[1]//2)

                        # x1 = random.randint(int(mid_x-ori_w*0.15),int(mid_x+ori_w*0.15)) - w//2
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap: assert(i.size == (w, h))

                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                else:
                    result = []

                    if random.random() < 0.5:
                        w, h = h, w

                    for idx, i in enumerate(imgmap):
                        if w <= img1.size[0] and h <= img1.size[1]:
                            if idx % self.clip_len == 0:
                                mid_x = int(img1.size[0]//2)

                                x1 = random.randint(int(mid_x-ori_w*0.15),int(mid_x+ori_w*0.15)) - w//2
                                y1 = random.randint(0, img1.size[1] - h)

                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert(result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [i.resize((self.size, self.size), self.interpolation) for i in result] 

            # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else: #don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size)
            return crop(imgmap)


class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None, clip_len=0):
        self.consistent = consistent
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5
        self.clip_len = clip_len
    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for idx, i in enumerate(imgmap):
                if idx % self.clip_len == 0: th = random.random()
                if th < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i) 
            assert len(result) == len(imgmap)
            return result 




class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=False, p=1.0, clip_len=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p 
        self.clip_len = clip_len

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)


        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold: # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                if self.clip_len == 0:
                    return [self.get_params(self.brightness, self.contrast, self.saturation, self.hue)(img) for img in imgmap]
                else:
                    result = []
                    for idx, img in enumerate(imgmap):
                        if idx % self.clip_len == 0:
                            transform = self.get_params(self.brightness, self.contrast,
                                                        self.saturation, self.hue)
                        result.append(transform(img))
                    return result

        else: # don't do ColorJitter, do nothing
            return imgmap 

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string



class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]

class ToPIL:
    def __call__(self, imgmap):
        topil = transforms.ToPILImage()
        return [topil(i) for i in imgmap]

class Normalize:
    def __init__(self, dataset=None,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        if 'diving' in dataset:
            self.mean = [0.3381, 0.5108, 0.5785]
            self.std = [0.2206, 0.2309, 0.2615]
        else:
            self.mean = mean
            self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')