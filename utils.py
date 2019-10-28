import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from datasets import ImageDataset,DescriptionDataset,TitleDataset
from models import MINet,DescriptionNet,TitleNet,get_pretrained_resnet

def get_dataset(name):
    if name == 'image':
        return ImageDataset(transform=get_augmentation_dict())
    elif name == 'title':
        return TitleDataset()
    elif name == 'description':
        return DescriptionDataset()
    else:
        print('Wrong dataset name!')

def get_model(name, ds):
    classes = len(ds.y_codec.classes_)
    if name == 'resnet_pretrained':
        return get_pretrained_resnet(classes)
    elif name == 'title':
        return TitleNet(input_size=ds.x_len,num_classes=classes)
    elif name == 'description':
        return DescriptionNet(input_size=ds.x_len,num_classes=classes)
    else:
        print('Wrong model name!')


def get_augmentation_dict():
    """Returns augmentations dictionary for ImageDataset (training and validation)"""
    return  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

def imshow(img):
    """Util function to plot image"""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def split_dataset(ds, val):
    train_size = int((1-val) * len(ds))
    test_size = len(ds) - train_size
    return torch.utils.data.random_split(ds, [train_size,test_size])