'''
@FileName   :train.py
@Description:训练模型
@Date       :2022/09/26 15:44:35
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

from torch.autograd import Variable
from model.model_utils import vgg16_bn
from model.model_utils import resnet50
from model.yoloLoss import yoloLoss
from dataset import yoloDataset
from model.visualize import Visualizer  # 暂时屏蔽
import numpy as np
import matplotlib.pyplot as plt
import logging

#配置基本日志设置
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别，可以选择DEBUG, INFO, WARNING, ERROR, CRITICAL
    datefmt='%Y-%m-%d %H:%M:%S',  #设置日期时间格式
    filename='yolov1.log',  #日志输出路径
    filemode='w'  #指定日志文件的模式(a是追加,w是覆盖)
)

# 创建一个日志记录器
logger = logging.getLogger('my_logger')

dtype = torch.bool

# 判定是否使用cuda
use_gpu = torch.cuda.is_available()

# 参数的管理
# 参数含义 (S,B,l_coord,l_noobj):
learning_rate = 0.001  #学习率
num_epochs = 100  #训练轮数
batch_size = 4  #批大小
num_workers = 2  #线程数
default_backbone = True  #默认backbone
criterion = yoloLoss(7, 2, 5, 0.5)
# file_root = './data/combine_doc/images/'相对路径
file_root = 'E:\\Source\\Github\\datasets\\yolov1\\combine_doc\\images\\'  #绝对路径

# backbone使用resnet还是使用VGG16--默认resnet

if default_backbone:
    net = resnet50()
else:
    net = vgg16_bn()

# resnet的分类器的结构组成

# net.classifier = nn.Sequential(
#                  nn.Linear(512 * 7 * 7, 4096),
#                  nn.ReLU(True),
#                  nn.Dropout(),
#                  nn.Linear(4096, 4096),
#                  nn.ReLU(True),
#                  nn.Dropout(),
#                  nn.Linear(4096, 1470), )

# net = resnet18(pretrained=True)
# net.fc = nn.Linear(512,1470)

# initial Linear

# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 0.01)
#         m.bias.data.zero_()
# net.load_state_dict(torch.load('yolo.pth'))

# 查看网络结构和加载的参数
# print(net)
# print('load pre-trined model')

# 判定使用的网络结构
# 加载对应的模型参数

if default_backbone:  # resnet
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        # print(k)
        if k in dd.keys() and not k.startswith('fc'):
            # print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)

else:  # vgg
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        # print(k)
        if k in dd.keys() and k.startswith('features'):
            # print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)

# if False:
# net.load_state_dict(torch.load('best.pth'))

print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

if use_gpu:
    net.cuda()

net.train()

# 原优化器参数--Lr的差异性
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr': learning_rate}]

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# 源路径
# train_dataset = yoloDataset(root=file_root, list_file=['voc2012.txt', 'voc2007.txt'],
#                             train=True, transform = [transforms.ToTensor()] )
# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',
#                            train=False,transform = [transforms.ToTensor()] )

# 训练数据
train_dataset = yoloDataset(root=file_root,
                            list_file='E:\\Source\\Github\\datasets\\yolov1\\combine_doc\\voc_ours_train.txt',
                            train=True,
                            transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# 测试数据
test_dataset = yoloDataset(root=file_root,
                           list_file='E:\\Source\\Github\\datasets\\yolov1\\combine_doc\\voc_ours_test.txt',
                           train=False,
                           transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看数据集中的图片数量

print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))

logfile = open('./data/log.txt', 'w')

# 使用时修改为自己的环境名： env='XXX'
# vis = Visualizer(env='py38_torch_gpu')
best_test_loss = np.inf


# 显示图片
def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    num_iter = 0
    for epoch in range(num_epochs):
        net.train()

        # 让学习率变化起来，使其更新到优化器中，使用时可解除注释
        # if epoch == 1:
        #     learning_rate = 0.0005
        # if epoch == 2:
        #     learning_rate = 0.00075
        # if epoch == 3:
        #     learning_rate = 0.001

        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001

        # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        for i, (images, target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)
            # total_loss += loss.data[0]
            total_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' %
                      (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data.item(), total_loss / (i + 1)))
                num_iter += 1
                # 暂时屏蔽训练可视化损失曲线
                # vis.plot_train_val(loss_train=total_loss / (i + 1))

        # validation
        validation_loss = 0.0
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images, volatile=True)
            target = Variable(target, volatile=True)
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.data.item()
        validation_loss /= len(test_loader)
        #屏蔽验证时的损失曲线
        # vis.plot_train_val(loss_val=validation_loss)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), './weights/best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()
        torch.save(net.state_dict(), './weights/yolo.pth')

# 在运行之前先打开terminal终端，输入以下命令，查看训练损失曲线。
# python -m visdom.server
