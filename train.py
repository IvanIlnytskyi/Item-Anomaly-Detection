from datasets import ImageDataset
from models import MINet
from torch import nn
import torch.optim as optim
import torch
from utils import get_augmentation_dict, split_dataset
from train_validate_utils import train

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
full_ds = ImageDataset(transform=get_augmentation_dict()['training'])
train_ds, val_ds = split_dataset(full_ds, 0.1)
net = MINet(len(full_ds.y_codec.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=16,
                                           shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=16,
                                           shuffle=True, num_workers=0)
#

config = {
    'device' : device,
    'epochs': 1,
    'net': net,
    'optimizer': optimizer,
    'train_loader': trainloader,
    'val_loader': valloader,
    'criterion': criterion
}

train(**config)