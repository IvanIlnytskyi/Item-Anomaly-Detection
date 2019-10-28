from torch import nn
import torch.optim as optim
import torch
from utils import get_augmentation_dict, split_dataset, get_dataset, get_model
from train_validate_utils import train
from tensorboardX import SummaryWriter
from optparse import OptionParser
import torch.utils.data


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-m', '--model', dest='model',
                      type='string', help='model')
    parser.add_option('-d', '--dataset', dest='dataset',
                      type='string', help='dataset')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='learning_rate', default=1e-4,
                      type='float', help='learning rate')
    parser.add_option('-s', '--train-val-split', dest='split', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-c', '--model-save-name', dest='model_name',
                      type='string', help='name of model for storing weights and logging')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    bs = args.batch_size
    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_ds = get_dataset(name=args.dataset)
    train_ds, val_ds = split_dataset(full_ds, args.split)
    net = get_model(name=args.model, ds=full_ds)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=bs,
                                              shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=bs,
                                            shuffle=True, num_workers=0)
    save_path = "../logs/"+model_name
    logger = SummaryWriter(save_path)
    model_path = "../models/"+model_name+'.pth'
    config = {
        'device': device,
        'epochs': args.epochs,
        'net': net,
        'optimizer': optimizer,
        'train_loader': trainloader,
        'val_loader': valloader,
        'criterion': criterion,
        'logger': logger,
        'model_path': model_path
    }

    train(**config)