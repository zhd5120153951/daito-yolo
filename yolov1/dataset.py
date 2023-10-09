'''
@FileName   :dataset.py
@Description:数据读取
@Date       :2022/09/27 10:19:26
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标(横向和纵向)
'''
import os
import sys
import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from model.model_utils import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# dtype=torch.bool


class yoloDataset(data.Dataset):
    image_size = 448

    def __init__(
        self,
        root,
        list_file,
        train,
        transform,
    ):
        print('data init')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB
        self.random_flip = random_flip
        self.randomScale = randomScale
        self.randomBlur = randomBlur
        self.RandomBrightness = RandomBrightness
        self.RandomHue = RandomHue
        self.RandomSaturation = RandomSaturation
        self.randomShift = randomShift
        self.randomCrop = randomCrop
        self.BGR2RGB = BGR2RGB
        self.subMean = subMean

        # 用指令合并数据集，报错就给注释了
        # if isinstance(list_file, list): # 判断list_file是一个list还是其他类型，如果是list就合并
        #     # Cat multiple list files together.
        #     # This is especially useful for voc07/voc12 combination.
        #
        #     # tmp_file = '/tmp/listfile.txt' # 这里的listfile是传入的文件名，和root是相对路关系
        #     # 用tmp_file进行读取一方修改到原始的文件
        #     #  file_root = './data/VOC0712/VOC2007/JPEGImages'  # 方便查看
        #
        #     tmp_file = './data/VOC0712/listfile.txt'
        #
        #     if not os.path.exists(tmp_file):
        #         os.mkdir(tmp_file)
        #
        #     # print('cat %s > %s' % (' '.join(list_file), tmp_file))
        #
        #     # os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
        #     os.system('type %s > %s' % (' '.join(list_file), tmp_file)) # 把voc2012和voc2007合并到listfile.txt
        #     list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            # print(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5  # 由于包含五个信息，所以每五个数就是一个box
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])  # split[0]是图片名，后面才是坐标（x,y,w,h）
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]  # 类别——class
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)  # 类0开是背景，所以要加1
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)  # 样本数等于框数

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # print(self.root + fname)

        img = cv2.imread(os.path.join(self.root + fname))
        # print(os.path.join(self.root + fname))
        # print("图像：", img)

        # cv2.imshow("img", img)
        # cv2.waitKey()

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            # img = self.random_bright(img)
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        # debug-删

        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()

        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)  # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels)  # 7x7x30
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        grid_num = 14
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target


# 测试是否能正常读取：
def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    # 源路径
    # file_root = '../data_dir/VOC0712/VOC2007/JPEGImages/'

    file_root = './data/combine_doc/images/'

    # 源路径
    # train_dataset = yoloDataset(root=file_root, list_file='voc12_trainval.txt',
    #                             train=True, transform = [transforms.ToTensor()] )

    # img = cv2.imread(os.path.join(self.root+fname))
    # list_file=[r'../data_dir/VOC0712/voc2012.txt', r'../data_dir/VOC0712/voc2007.txt']

    train_dataset = yoloDataset(root=file_root,
                                list_file='./data/combine_doc/voc2012.txt',
                                train=True,
                                transform=[transforms.ToTensor()])

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)
    train_iter = iter(train_loader)

    # print(train_iter)

    for i in range(100):
        img, target = next(train_iter)
        print(img, target)


if __name__ == '__main__':
    main()
    print("验证结束...")
