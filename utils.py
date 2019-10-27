import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
def get_augmentation_dict():
    """Returns augmentations dictionary for ImageDataset (training and validation)"""
    return {
        'training': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

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