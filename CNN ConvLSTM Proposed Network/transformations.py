'''
These transformations are used from PyTorch Vision transformations From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
'''
import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None


#Apply multiple actions/transformations at the same pipeline
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, inv=False, flow=False):
        for t in self.transforms:
            img = t(img, inv, flow)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

#convert numpy array to tensor
class ToTensor(object):
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic, inv, flow):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass

#Normalization of images by subtracting mean and dividing by standard deviation
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, inv):
        mean = self.mean
        std = self.std
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass

# Rescaling of Images
class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, inv, flow):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass

#Cropping the center of the image
class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, inv, flow):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass

#Horizontal Flip of the images
class RandomHorizontalFlip(object):

    def __call__(self, img, inv, flow):
        if self.p < 0.5:
            img =  img.transpose(Image.FLIP_LEFT_RIGHT)
            if inv is True:
                img = ImageOps.invert(img)
        return img

    def randomize_parameters(self):
        self.p = random.random()

#Corner Crop of the frame at different scales + resizing
class MultiScaleCornerCrop(object):
    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img, inv, flow):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 1
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 1
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]

#Flip images horizontally
class FlippedImagesTest(object):
    def __init__(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.mean = mean
        self.std = std
        self.to_Tensor = ToTensor()
        self.normalize = Normalize(self.mean, self.std)

    def __call__(self, img, inv=False, flow=False):
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        if inv is True:
            img_flipped = ImageOps.invert(img_flipped)

        tensor_img = self.to_Tensor(img, inv, flow)
        tensor_img_flipped = self.to_Tensor(img_flipped, inv, flow)

        normalized_img = self.normalize(tensor_img, inv, flow)
        normalized_img_flipped = self.normalize(tensor_img_flipped, inv, flow)
        horFlippedTest_imgs = [normalized_img, normalized_img_flipped]
        horFlippedTest_imgs = torch.stack(horFlippedTest_imgs, 0)
        return horFlippedTest_imgs

    def randomize_parameters(self):
        pass