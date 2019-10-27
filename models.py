from torch import nn
import torch.nn.functional as F


def conv_module(in_num, out_num):
    """"Creates convolution block"""
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))


def fully_connected_module(in_num, out_num):
    """"Creates fully connected layer with relu and dropout """
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_num, out_num),
        nn.ReLU(inplace=True)
    )


class MINet(nn.Module):
    def __init__(self, num_classes):
        super(MINet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = conv_module(3, 6)
        self.layer2 = conv_module(6, 12)
        self.layer3 = conv_module(12, 18)
        self.layer4 = conv_module(18, 24)
        self.layer5 = conv_module(24, 32)
        self.layer6 = conv_module(32, 48)

        self.fcm1 = fully_connected_module(48*3*3, 100)
        self.fcm2 = fully_connected_module(100, 100)
        self.fcm3 = nn.Linear(100, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(-1, 48 * 3 * 3)
        out = self.fcm1(out)
        out = self.fcm2(out)
        out = self.fcm3(out)

        return out

    class TitleNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MINet, self).__init__()
            self.num_classes = num_classes
            self.input_size = input_size

            self.fcm1 = fully_connected_module(input_size, 400)
            self.fcm2 = fully_connected_module(400, 120)
            self.fcm3 = fully_connected_module(120, 20)
            self.fcm4 = nn.Linear(50, num_classes)

        def forward(self, x):
            out = self.fcm1(x)
            out = self.fcm2(out)
            out = self.fcm3(out)
            out = self.fcm4(out)

            return out

        class DescriptionNet(nn.Module):
            def __init__(self, input_size, num_classes):
                super(MINet, self).__init__()
                self.num_classes = num_classes
                self.input_size = input_size

                self.fcm1 = fully_connected_module(input_size, 100)
                self.fcm2 = fully_connected_module(100, 80)
                self.fcm3 = fully_connected_module(80, 40)
                self.fcm4 = nn.Linear(40, num_classes)

            def forward(self, x):
                out = self.fcm1(x)
                out = self.fcm2(out)
                out = self.fcm3(out)
                out = self.fcm4(out)

                return out

