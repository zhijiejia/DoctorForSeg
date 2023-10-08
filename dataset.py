import os
import glob
import torch
import cv2
import math
import numpy as np
from PIL import Image
from random import random, randint
import numpy as np


class SegDataset(torch.utils.data.Dataset):

    '''
        segmentation dataset
    '''

    def __init__(self,
                 transforms,
                 mode='train'):

        self.transforms = transforms
        self.image_list = []
        self.label_list = []
        self.mode = mode
        self.name = 'Voc'

        self.image_root = '/home/jzj/spixel_guided_segmentation-main/VOCdevkit/VOC2012'
        lst = open(f'{self.image_root}/ImageSets/Segmentation/{self.mode}.txt', mode='r', encoding='utf-8').readlines()
        self.image_list = self.image_list + lst

        print(f'Load {self.mode} Dataset, Total {len(self.image_list)} !')

    def __getitem__(self, idx):
        image_name = self.image_list[idx].strip()
        image_path = f'/home/jzj/spixel_guided_segmentation-main/VOCdevkit/VOC2012/JPEGImages/{image_name}.jpg'
        label_path = f'/home/jzj/spixel_guided_segmentation-main/VOCdevkit/VOC2012/SegmentationClassAug/{image_name}.png'

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        image, label = self.transforms(image, label)
        
        return image, label, image_name

    def __len__(self):
        return len(self.image_list)
    

class CityscapeDataset(torch.utils.data.Dataset):

    '''
        cityscape segmentation dataset
    '''

    def __init__(self,
                 transforms,
                 mode='train'):

        self.transforms = transforms
        self.image_list = []
        self.mode = mode
        self.name = 'Cityscape'

        lst = glob.glob(f'cityscape/leftImg8bit/{mode}/*/*.png')
        self.image_list = self.image_list + lst

        print(f'Load {self.mode} Dataset, Total {len(self.image_list)} !')

    def __getitem__(self, idx):
        image_path = self.image_list[idx].strip()
        image_name = '_'.join(image_path.split('/')[-1].split('_')[:-1])
        label_path = image_path.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        output = self.transforms(image=image, mask=label)
        image, label = output['image'], output['mask']
        
        return image, label, image_name

    def __len__(self):
        return len(self.image_list)
    

class BirdDataset(torch.utils.data.Dataset):

    '''
        bird segmentation dataset
    '''

    def __init__(self,
                 transforms,
                 mode='train'):

        self.transforms = transforms
        self.image_list = []
        self.mode = mode
        self.name = 'bird'

        lst = open(f'bird/{mode}.txt', mode='r', encoding='utf-8').readlines()
        self.image_list = self.image_list + lst

        print(f'Load {self.mode} Dataset, Total {len(self.image_list)} !')

    def __getitem__(self, idx):
        image_path = self.image_list[idx].strip()
        image_name = image_path.split('/')[-1].split('.')[0]
        label_path = image_path.replace('/images/', '/BinaryMask/').replace('.jpg', '.png')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label[label > 0] = 1

        output = self.transforms(image=image, mask=label)
        image, label = output['image'], output['mask']
        
        return image, label, image_name

    def __len__(self):
        return len(self.image_list)