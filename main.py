import torch
from Network import AlexNet, AlexNetFCN, vgg16Fcn
from GoogleNet import GoogLeNet, googLeNet
from Data import LoadData
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import re
from torchvision.transforms import ToPILImage
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)
best_wts_name = 'best_wts3.pt'

# 计算损失（误差）
def _loss(x_class, label_class, criterion, Config):
    loss = None
    if len(x_class.shape) == 4:
        loss = criterion(x_class, label_class)
    else:
        criterion = torch.nn.MSELoss()
        label = label_class.view(-1, label_class.shape[-1] * label_class.shape[-2])
        x_class[x_class < 0.5] = 0
        x_class[x_class >= 0.5] = 1
        loss = criterion(x_class, label)
    return loss

# 计算准确率
def _accuracy(x_class, label_class, threhold):
    cnt = 0
    if len(x_class.shape)==4:
        # pre_class = torch.clone(x_class)
        pre_class = torch.argmax(x_class, 1)
        cnt = torch.sum(pre_class == label_class)
    else:
        label = label_class.view(-1, label_class.shape[-1] * label_class.shape[-2])
        x_class[x_class<0.5]=0
        x_class[x_class>=0.5]=1
        cnt = torch.sum(x_class==label)
    return cnt


# 训练
def train(model, data_loader, criterion, optimizer, Config, device, start_epoch=1, best_acc=0):
    fig = plt.figure()
    ax_color = ('#337ab7', '#5cb85c', '#d9534f')  # 训练集、测试集和最佳模型颜色
    ax_loss, ax_acc = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    ax_loss.set_title('Loss')
    ax_acc.set_title('Accuracy')
    ax_acc.set_ylim(0, 1)  # 准确率的范围为0-1
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    plt.ion()

    # train_max_acc, test_max_acc = []
    best_acc = best_acc
    best_wts = dict()
    for epoch in range(start_epoch, Config['epochs'] + 1):
        for phase_id, phase in enumerate(('train', 'test')):
            epoch_loss = 0
            if phase == 'train':
                model.train()
                loss_list = train_loss
                acc_list = train_acc
            else:
                model.eval()
                loss_list = test_loss
                acc_list = test_acc
            loader = data_loader[phase]
            acc_cnt = 0.
            for batch in loader:
                imgs = batch['image'].to(device)
                label_class = batch['label'].to(device)
                optimizer.zero_grad()    # 梯度清零

                with torch.set_grad_enabled(phase == 'train'):
                    x_class = model(imgs)   # 模型预测

                    pre_class = _accuracy(x_class, label_class, Config['pre_threhold'])
                    acc_cnt += pre_class
                    loss = _loss(x_class, label_class, criterion, Config)
                    print(f"batch: loss: {loss.item()}, acc: {float(acc_cnt)/(Config['output'][0] * Config['output'][1])}")
                    epoch_loss += loss.item()

                    if phase == 'train':
                        loss.backward()    # 损失回传
                        optimizer.step()    # 梯度更新

            epoch_loss /= len(loader)
            acc = acc_cnt / (len(loader.dataset)* Config['output'][0] * Config['output'][1]) * 100.
            # 绘图
            plt.cla()
            loss_list.append(epoch_loss)
            acc_list.append(acc)
            ax_acc.plot(range(start_epoch, epoch + 1), acc_list, label=phase, c=ax_color[phase_id])
            ax_loss.plot(range(start_epoch, epoch + 1), loss_list, label=phase, c=ax_color[phase_id])
            if epoch==start_epoch:
                ax_acc.legend(loc='upper left')
                ax_loss.legend(loc='upper right')
            print(f"Epoch: {epoch}, {phase} loss: {epoch_loss}, acc: {acc:.2f}%")

            if phase == 'test' and acc > best_acc:
                best_acc = acc
                # plt.text(epoch, acc + 0.01, '{acc:.2f}%', c=ax_color[2])  # 标注当前最佳模型
                best_wts = deepcopy(model.state_dict())
                torch.save(best_wts, best_wts_name)
                print(f"Best acc is {best_acc}, saved model")
            elif phase == 'train' and epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(Config['model_dir'], f'model_{epoch}.pt'))
                print(f"saved model model_{epoch}.pt")
        print("==================================================")
    plt.ioff()
    plt.show()
    return best_acc


# 测试
def test(model, data_loader, criterion, device, config):
    # last model
    model_name = sorted([x for x in os.listdir(config['model_dir']) if x.startswith('model')],
                        key=lambda x: int(re.match('model_([\d]+).pt', x).group(1)))[-1]
    start_epoch = int(re.match('model_([\d]+).pt', model_name).group(1))
    try:
        model.load_state_dict(torch.load(os.path.join(config['model_dir'], model_name)))
    except RuntimeError:
        model.load_state_dict(
            torch.load(os.path.join(config['model_dir'], model_name), map_location=lambda storage, loc: storage))
    print(f"Load lsat model: {model_name}")

    acc_cnt = 0.
    total_loss = 0.
    for phase in ('train', 'test'):
        loader = data_loader[phase]
        acc_list = []
        loss_list = []
        for idx, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            label_class = batch['label'].to(device)

            with torch.no_grad():
                x_class = model(imgs)

                acc_cnt = _accuracy(x_class, label_class, config['pre_threhold'])
                loss = _loss(x_class, label_class, criterion, config)
                loss_list.append(loss.item())
                batch_acc = float(acc_cnt)/(config['output'][0] * config['output'][1]*len(imgs))
                # print(f"batch {idx}, acc: {batch_acc}")
                acc_list.append(batch_acc)

        phase_loss = np.mean(loss_list)
        phase_acc = np.mean(acc_list)
        print(f"{phase} loss: {phase_loss}, accuracy: {phase_acc}")


# 断点训练
def finetine(model, data_loader, criterion, optimizer, Config, device):
    best_acc = 0
    best_wts = dict()
    # best model
    # if os.path.exists(best_wts_name):
    #     # 加载最佳模型
    #     try:
    #         model.load_state_dict(torch.load(best_wts_name))
    #     except RuntimeError:
    #         model.load_state_dict(torch.load(best_wts_name, map_location=lambda storage, loc: storage))
    #     # 计算最佳准确率
    #     loader = data_loader['test']
    #     model.eval()
    #     with torch.no_grad():
    #         acc_cnt = 0
    #         for idx, batch in enumerate(loader):
    #             imgs = batch['image'].to(device)
    #             label_class = batch['label'].to(device)
    #             x_class = model(imgs)
    #             # loss = _loss(x_class, label_class, criterion)
    #             pred_class = _accuracy(x_class, label_class, Config['pre_threhold'])
    #             acc_cnt += pred_class
    #     acc = acc_cnt.float() / (len(loader.dataset)* Config['output'][0] * Config['output'][1]) * 100.
    #     print("Best accuracy is ", acc)
    #     print("Load best wts")
    #     best_acc = acc

    # last model
    model_name = sorted([x for x in os.listdir(Config['model_dir']) if x.startswith('model')], key=lambda x: int(re.match('model_([\d]+).pt', x).group(1)))[-1]
    start_epoch = int(re.match('model_([\d]+).pt', model_name).group(1))
    try:
        model.load_state_dict(torch.load(os.path.join(Config['model_dir'], model_name)))
    except RuntimeError:
        model.load_state_dict(torch.load(os.path.join(Config['model_dir'], model_name), map_location=lambda storage, loc: storage))
    print(f"Load lsat model: {model_name}, continue train from {start_epoch} to {Config['epochs']}")

    acc = train(model, data_loader, criterion, optimizer, Config, device, start_epoch, best_acc)
    return acc


# 预测
def predict(model, data_loader, Config, device):
    try:
        model.load_state_dict(torch.load(best_wts_name))
    except RuntimeError:
        model.load_state_dict(torch.load(best_wts_name, map_location=lambda storage, loc: storage))
    print("Load model: ", best_wts_name)
    # last model
    # model_name = sorted([x for x in os.listdir(Config['model_dir']) if x.startswith('model')],
    #                     key=lambda x: int(re.match('model_([\d]+).pt', x).group(1)))[-1]
    # start_epoch = int(re.match('model_([\d]+).pt', model_name).group(1))
    # try:
    #     model.load_state_dict(torch.load(os.path.join(Config['model_dir'], model_name)))
    # except RuntimeError:
    #     model.load_state_dict(
    #         torch.load(os.path.join(Config['model_dir'], model_name), map_location=lambda storage, loc: storage))
    # print(f"Load lsat model: {model_name}")

    loader = data_loader['test']
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            label_class = batch['label'].to(device)
            x_class = model(imgs)
            # 展示一个batch
            if idx==0:
                for i in range(len(imgs)):
                    plt.subplot(1, 3, 1)
                    plt.title("src")
                    plt.imshow(ToPILImage()(imgs[i]))
                    plt.subplot(1, 3, 2)
                    plt.title("label")
                    label_img = label_class[i].float()
                    plt.imshow(label_img)
                    plt.subplot(1, 3, 3)
                    plt.title("predict")
                    pre_img = x_class[i].float()
                    if len(x_class.shape)==4:
                        pre_img = torch.argmax(pre_img, 0)
                        pre_img[pre_img==1] = 255
                        plt.imshow(pre_img.float())
                    else:
                        pre_img[pre_img < 0.5] = 0
                        pre_img[pre_img >= 0.5] = 1
                        plt.imshow(pre_img.reshape(50, 50).float().cpu())
                    plt.show()
            break




def main(phase='train', model=None):
    print("==> Load data")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    Config = model.getConfig()    #　获取配置
    Config['epochs']=1000
    loader = LoadData(Config)
    print("==> Build model")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['lr'])
    if phase.capitalize() == 'Train':
        print("===> Start Train")
        acc = train(model, loader, criterion, optimizer, Config, device)
        print("Best accuracy is: ", acc)
    elif phase.capitalize() == 'Finetune':
        print("==> Start Finetune")
        acc = finetine(model, loader, criterion, optimizer, Config, device)
        print("Best accuracy is: ", acc)
    elif phase.capitalize() == 'Predict':
        print("==> Start Predict")
        predict(model, loader, Config, device)
    elif phase.capitalize() == 'Test':
        print("==> Start Test")
        test(model, loader, criterion, device, Config)


if __name__ == '__main__':
    phase = 'predict'
    # best_wts_name = 'trained_model/vgg16_300.pt'
    # model = googLeNet(2)
    # model = vgg16Fcn(num_classes=2)

    # AlexNet 回归
    # best_wts_name = 'trained_model/AlexNet.pt'
    # model = AlexNet()

    # AlexNetFCN 全卷积
    # best_wts_name = 'trained_model/AlexNetFCN.pt'
    # model = AlexNetFCN(num_classes=2)    # alexnet fcn

    # GoogLeNetFCN 全卷积
    best_wts_name = 'trained_model/GoogLeNetFCN.pt'
    model = GoogLeNet(num_classes=2)    # googlenet fcn

    main(phase, model)
    # acc_list = cleanData()