import torch
from torch import nn
from Config import Config
import numpy as np


# 基础:回归
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.config = Config.copy()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48 * 2, 11, stride=4, padding=0),
            nn.BatchNorm2d(48 * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48 * 2, 128 * 2, 5, 1, 2),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 * 2, 192 * 2, 3, 1, 1),
            nn.BatchNorm2d(192 * 2),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192 * 2, 192 * 2, 3, 1, 1),
            nn.BatchNorm2d(192 * 2),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192 * 2, 128 * 2, 3, 1, 1),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 128 * 2, 2048 * 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(2048 * 2, 2048 * 2),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048 * 2, Config['output'][0] * Config['output'][1])
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 6 * 6 * 128 * 2)
        x_class = self.classifier(x)
        return x_class.float()

    def getConfig(self):
        self.config['train_size'] = (227, 227)
        self.config['output'] = (50,50)
        return self.config


# 双线性插值
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


# 全卷积
class AlexNetFCN(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetFCN, self).__init__()
        self.config = Config.copy()
        # pretrained_net = AlexNet(pretrained=True)
        # pretrained_net = AlexNet()
        # self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-1])  # 第一段
        # self.stage2 = list(pretrained_net.children())[:3]  # 第二段
        # self.stage3 = list(pretrained_net.children())[:2]  # 第三段
        # AlexNet
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=0),    # 58*58*96
            nn.BatchNorm2d(48 * 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),    # 57*57*256
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),    # 28*28*256
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 * 2, 192 * 2, 3, 1, 1),    # 28*28*384
            nn.BatchNorm2d(192 * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),    # 14*14*384
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192 * 2, 192 * 2, 3, 1, 1),    # 14*14*384
            nn.BatchNorm2d(192 * 2),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192 * 2, 128 * 2, 3, 1, 1),    # 14*14*256
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, ceil_mode=True),    # 7*7*256
        )
        # 通道统一
        self.scores1 = nn.Conv2d(256, num_classes, 1)
        self.scores2 = nn.Conv2d(384, num_classes, 1)
        self.scores3 = nn.Conv2d(256, num_classes, 1)

        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        # 2倍上采样
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # x = self.stage1(x)
        # s1 = x  # 1/8
        #
        # x = self.stage2(x)
        # s2 = x  # 1/16
        #
        # x = self.stage3(x)
        # s3 = x  # 1/32
        s1 = conv5
        s2 = conv4
        s3 = conv2

        s1 = self.scores1(s1)
        s1 = self.upsample_2x(s1)  # 1/16
        s2 = self.scores2(s2)
        s2 = s2 + s1

        s3 = self.scores3(s3)
        s2 = self.upsample_4x(s2)  # 1/8
        s = s3 + s2

        s = self.upsample_8x(s)  # 1/1
        s = self.softmax(s).float()
        return s

    def getConfig(self):
        self.config['train_size'] = (240, 240)
        self.config['output'] = (224, 224)
        self.config['train_batch_size'] = 30
        self.config['test_batch_size'] = 10
        return self.config


class vgg16Fcn(nn.Module):
    def __init__(self, num_classes):
        super(vgg16Fcn, self).__init__()
        self.num_classes = num_classes
        self.config = Config.copy()
        #  cvon1  input:224*224*3  output: 112*112*64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 112*112*64
        )

        # conv2    # 56*56*128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 56*56*128
        )

        # conv3     # 28*28*256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # conv4     # 14*14*512
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # conv5     # 7*7*512
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(256, num_classes, 1)

        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        # 2倍上采样
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        s1 = conv5
        s2 = conv4
        s3 = conv3

        s1 = self.scores1(s1)
        s1 = self.upsample_2x(s1)  # 1/16
        s2 = self.scores2(s2)
        s2 = s2 + s1

        s3 = self.scores3(s3)
        s2 = self.upsample_4x(s2)  # 1/8
        s = s3 + s2

        s = self.upsample_8x(s)  # 1/1
        s = self.softmax(s).float()
        return s

    def getConfig(self):
        self.config['train_size'] = (224, 224)
        self.config['output'] = (224, 224)
        self.config['train_batch_size'] = 30
        self.config['test_batch_size'] = 10
        return self.config


