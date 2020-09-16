from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from Config import Config

label_dir = 'label'
# src_dir = 'src'
src_dir = 'EnSrc'

def idea():
    img = cv2.imread("label/1_0.tif", 0)
    size = img.shape
    resize_img = cv2.resize(img, (50, 50))
    plt.subplot(3,1,1)
    plt.imshow(resize_img, cmap='gray')
    re_resize_img = cv2.resize(resize_img, size)
    plt.subplot(3,1,2)
    plt.imshow(re_resize_img, cmap='gray')
    re_resize_img[re_resize_img>0] = 255
    plt.subplot(3,1,3)
    plt.imshow(re_resize_img, cmap='gray')
    plt.show()
    max_acc = np.sum(re_resize_img.flatten() == img.flatten()) / img.flatten().shape
    print("Max accuracy: %.2f%%" % (max_acc*100))


class MyDataset(Dataset):
    def __init__(self, src, label, idxs, transform, Config):
        self.data = np.array( [os.path.join(src, x) for x in os.listdir(src)] )[idxs]
        self.label = np.array( [os.path.join(label, x) for x in os.listdir(src)] )[idxs]
        self.transform = transform
        self.Config = Config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        image = self.transform(image)    # 预处理
        label = Image.open(self.label[idx]).convert('L')    # 转成灰度图
        label_old_size = label.size
        # label = np.array( label.resize(Config['output']) ).flatten()
        label = np.array( label.resize(self.Config['output']) )
        label[label!=0]=1
        # label = np.array( label.resize(image.shape[1:]) )
        label = torch.from_numpy(label)
        sample = {'image': image, 'label': label.float(), 'label_old_size': label_old_size, 'filename': self.data[idx]}
        return sample


def LoadData(Config):
    data_len = len(os.listdir(src_dir))
    # 训练集、测试集82分
    train_idx = np.random.choice(np.arange(data_len), int(data_len*0.8), replace=False)
    test_idx = np.setdiff1d(np.arange(data_len), train_idx)
    # 数据的预处理
    train_transform = transforms.Compose([
        transforms.Resize(Config['train_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(Config['train_size']),
        transforms.ToTensor(),
    ])

    train_dataset = MyDataset(src_dir, label_dir, train_idx, train_transform, Config)
    test_dataset = MyDataset(src_dir, label_dir, test_idx, train_transform, Config)
    # 打包批次
    train_loader = DataLoader(train_dataset, batch_size=Config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config['test_batch_size'], shuffle=True)
    loader = {'train': train_loader, 'test': test_loader}
    return loader


def LabelBackSize(label, old_size):
    label = Image.fromarray(label.detach().numpy().reshape(Config['output']))
    label = label.resize(old_size)
    return label


def green(loader):
    dataset = loader['train'].dataset
    show_cnt = 3
    for i in np.random.choice(len(dataset), show_cnt, replace=False):
        img = dataset[i]['image']
        label = dataset[i]['label']
        plt.subplot(1, 3, 1)
        plt.imshow(transforms.ToPILImage()(img))
        green_img = img
        loc = [green_img[0]-green_img[1]-green_img[2]>0]
        green_img[1][loc] += (1-green_img[1][loc]) * 0.9
        plt.subplot(1, 3, 2)
        plt.imshow(transforms.ToPILImage()(green_img))
        plt.subplot(1, 3, 3)
        plt.imshow(label)
        plt.show()

        print(i)



if __name__ == '__main__':
    # idea()
    loader = LoadData(Config)
    green(loader)
    # train_loader = loader['train']
    # show_cnt = 5
    # for idx in np.random.randint(0, len(train_loader), show_cnt):
    #     sample = train_loader.dataset[idx]
    #     image = sample['image']
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(transforms.ToPILImage()(image))
    #     label = LabelBackSize( sample['label'], sample['label_old_size'])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(label, cmap='gray')
    #     plt.show()
