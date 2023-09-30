""" Code in this file is adpated from rpmcruz/autoaugment
https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

Heavy Data Augmentation Operations.

Transformation operations for Images. PIL.Image is used in all of them.
"""
import random

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps


def AutoContrast(img, _):
    """
    Apply automatic contrast on the given image.
    :param img: PIL Image object
    :return: PIL Image object with enhanced contrast
    """
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    """
    Invert the colors of the given image.
    :param img: PIL Image object
    :return: PIL Image object with inverted colors
    """
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    """
    Equalize the histogram of the given image.
    :param img: PIL Image object
    :return: PIL Image object with equalized histogram
    """
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):
    """
    Solarize the given image.
    :param img: PIL Image object
    :param v: Threshold, integer in the range [0, 256]
    :return: PIL Image object solarized
    """
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):
    """
    Posterize the given image.
    :param img: PIL Image object
    :param v: Bits to use, integer in the range [4, 8]
    :return: PIL Image object posterized
    """
    assert 4 <= v <= 8
    return PIL.ImageOps.posterize(img, int(v))


def Posterize2(img, v):
    """
    Posterize the given image.
    :param img: PIL Image object
    :param v: Bits to use, integer in the range [0, 4]
    :return: PIL Image object posterized
    """
    assert 0 <= v <= 4
    return PIL.ImageOps.posterize(img, int(v))


def Contrast(img, v):
    """
    Adjust the contrast of the given image.
    :param img: PIL Image object
    :param v: Contrast factor, float in the range [0.1, 1.9]
    :return: PIL Image object with adjusted contrast
    """
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    """
    Adjust the color of the given image.
    :param img: PIL Image object
    :param v: Color adjustment factor, float in the range [0.1, 1.9]
    :return: PIL Image object with adjusted color
    """
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    """
    Adjust the brightness of the given image.
    :param img: PIL Image object
    :param v: Brightness factor, float in the range [0.1, 1.9]
    :return: PIL Image object with adjusted brightness
    """
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    """
    Adjust the sharpness of the given image.
    :param img: PIL Image object
    :param v: Sharpness factor, float in the range [0.1, 1.9]
    :return: PIL Image object with adjusted sharpness
    """
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    """
    Apply absolute cutout augmentation to the given image.
    :param img: PIL Image object
    :param v: Cutout size, float value
    :return: PIL Image object with a region cut out
    """


def Cutout(img, v):
    """
    Apply cutout augmentation to the given image using a specified percentage.
    :param img: PIL Image object
    :param v: Cutout percentage, float in the range [0, 0.2]
    :return: PIL Image object with a region cut out
    """
    assert 0.0 <= v <= 0.2
    if v <= 0.0:
        return img

    v = v * img.size[0]

    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


class RandomAugmentation(object):
    """
    Apply a subset of random augmentation policies from a set of random transformations.
    """

    @staticmethod
    def augment_list():
        """
        Get the list of available augmentations with their corresponding function, min and max values.
        :return: List of tuples containing (function, min_value, max_value)
        """
        l = [
            (AutoContrast, 0, 1),
            (Invert, 0, 1),
            (Equalize, 0, 1),
            (Solarize, 0, 256),
            (Posterize, 4, 8),
            (Contrast, 0.1, 1.9),
            (Color, 0.1, 1.9),
            (Brightness, 0.1, 1.9),
            (Sharpness, 0.1, 1.9),
            (Cutout, 0, 0.2),
            (Posterize2, 0, 4),
        ]
        return l

    def __init__(self, policies):
        self.policies = policies
        self._augment_dict = {
            fn.__name__: (fn, v1, v2)
            for fn, v1, v2 in RandomAugmentation.augment_list()
        }

    def __call__(self, img, num_augments=3):
        for _ in range(num_augments):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = self._apply_augment(img, name, level)
        return img

    def _apply_augment(self, img, name, level):
        """
        Apply the augmentation named 'name' with intensity level 'level' to the image 'img'.
        :param img: PIL Image object
        :param name: String, name of the augmentation to apply
        :param level: Float, level of augmentation intensity
        :return: PIL Image object after applying augmentation
        """
        augment_fn, low, high = self._augment_dict.get(name)
        return augment_fn(img.copy(), level * (high - low) + low)
