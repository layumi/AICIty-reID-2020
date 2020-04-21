"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torchvision import datasets
import os
import numpy as np
import random

class AugFolder(datasets.ImageFolder):

    def __init__(self, root, transform, transform2):
        super(AugFolder, self).__init__(root, transform, transform2)
        self.transform2 = transform2

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample2 = sample.copy() 
        if self.transform is not None:
            sample = self.transform(sample)
            sample2 = self.transform2(sample2)
        return sample, sample2, target
