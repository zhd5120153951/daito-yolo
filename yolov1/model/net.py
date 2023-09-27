'''
@FileName   :net.py
@Description:网络模型构建部分
@Date       :2022/09/26 16:38:39
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
from model.model_utils import *

# import sys
# print(sys.path)
# # sys.path.append('A:\Learning_doc\detection_learning\yolov1\code_dir\model\model_utils.py')

# backbone的类型
__all__ = [
    'VGG',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19_bn',
    'vgg19',
]

# 网络参数
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 定于使用的VGG网络
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, image_size=448):
        super(VGG, self).__init__()
        self.features = features  # 特征提取层
        self.image_size = image_size  # 影像尺寸

        # 定义的后面的分类网络,层数有略微区别
        """
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        # if self.image_size == 448:
        #     self.extra_conv1 = conv_bn_relu(512,512)
        #     self.extra_conv2 = conv_bn_relu(512,512)
        #     self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        """
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1470),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # if self.image_size == 448:
        #     x = self.extra_conv1(x)
        #     x = self.extra_conv2(x)
        #     x = self.downsample(x)
        x = x.view(x.size(0), -1)  # 转>batch、imgfeature
        x = self.classifier(x)
        x = F.sigmoid(x)  # 激活函数
        x = x.view(-1, 7, 7, 30)
        return x

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ----------------------------------------------------------------------------------------------------------------------
# 构建网络模型
# ----------------------------------------------------------------------------------------------------------------------
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag = True
    for v in cfg:
        s = 1
        if (v == 64 and first_flag):
            s = 2
            first_flag = False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------------------------------------------------
# 定义在网络构建中反复使用到的卷积模块
# ----------------------------------------------------------------------------------------------------------------------
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                         nn.BatchNorm2d(out_channels), nn.ReLU(True))


# ----------------------------------------------------------------------------------------------------------------------
# 测试网络
# ----------------------------------------------------------------------------------------------------------------------
def test():
    import torch
    from torch.autograd import Variable
    model = vgg16()  # 选择backbone——vgg的类型
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1470),
    )

    print(model.classifier[6])
    print(model)

    # img = torch.rand(2, 3, 224, 224)
    img = torch.randn((2, 3, 448, 448))
    img = Variable(img)
    output = model(img)
    print(output.size())


# 测试网络是否能跑通
if __name__ == '__main__':

    test()
