'''
@FileName   :predict.py
@Description:预测单张图像
@Date       :2022/09/27 10:17:15
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import torch
from torch.autograd import Variable

from model.model_utils import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

#  数据集中的种类数

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

#  每个种类对应的检测框颜色--一共21种==(包含背景1种+物体种类20种）

Color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
         [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
         [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

#  每个种类对应的检测框颜色


def decoder(pred):
    '''
    pred (tensor) 1x7x7x30 —— [batch, s*s, B]---B[class, box1, box2]--box1[p,x1,y1,x2,y2]
                                                          30                  5
    # 看后面的组成：这里应该是前半部分是  box1~5，box2~5，classes~20

    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14  # 7*7的网格，每个网格中两个检测框，所以每行每列 14 个
    boxes = []  # 方框
    cls_indexs = []  # 种类索引
    probs = []  # 预测框
    cell_size = 1. / grid_num  # 方格尺寸, 一张图像分为7*7的网格，第一个中心点就是H/s/2 = h/14

    pred = pred.data
    pred = pred.squeeze(0)  # 7x7x30

    contain1 = pred[:, :, 4].unsqueeze(2)  # box1 的置信度
    contain2 = pred[:, :, 9].unsqueeze(2)  # box2 的置信度
    contain = torch.cat((contain1, contain2), 2)

    # 对每个ceil的两个框共2×7×7个框进行筛选，取置信度（是否有物体）大于0.1的，并且一个ceil中有两个大于0.1框的取得分较大的那个。

    mask1 = contain > 0.1  # 大于阈值,返回bool类型的值，true，false，0-1
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it>0.9
    mask = (mask1 + mask2).gt(0)  # 比较函数， 得到a中比b中元素大的位置，大于为1，小于为0
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框

    # 将筛选后的框取置信度（概率最大的类别的概率）大于0.1的，然后转换为(x1,y1,x2,y2)的格式。

    for i in range(grid_num):  # 行数
        for j in range(grid_num):  # 列数
            for b in range(2):  # 两个框
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:  # 存在检测目标
                    # print(i, j, b)
                    box = pred[i, j, b * 5:b * 5 + 4]  # box的选取
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])  #

                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell---中心坐标

                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # convert[cx,cy,w,h] to [x1,y1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        # print(cls_index)

                        # 这里转化为torch.Tensor([])的形式
                        # 增加：cls_index = torch.Tensor([cls_index])

                        cls_index = torch.Tensor([cls_index])
                        # print(cls_index)
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)
                        '''
                        # 通过打印出来可以看到：
                        # cls_indexs1[tensor(14), tensor(14), tensor(14), tensor(14), tensor(16), tensor(16), tensor(16)
                        #           , tensor(16), tensor(14), tensor(14), tensor(16), tensor(16), tensor(11), tensor(11)
                        #           , tensor(11), tensor(11), tensor(11), tensor(11)] 是这样的形式，列表里面是tensor
                        # https://blog.csdn.net/weixin_45928096/article/details/123630375?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165838968216782184613881%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165838968216782184613881&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-123630375-null-null.142^v33^pc_search_v2,185^v2^control&utm_term=%20%20%20%20cls_indexs%20%3D%20torch.cat%28cls_indexs%2C0%29%20%23%28n%2C%29%20RuntimeError%3A%20zero-dimensional%20tensor%20%28at%20position%200%29%20cannot%20be%20concatenated&spm=1018.2226.3001.4187
                        对比可以看出，如果使用cat这个函数，就必须是具有维度的tensor，所以要么在这里把 cls_index 转化问题tensor（[x],[x],[x]...）的形式
                        要么就从源头输出的时候,把单个的cls_index转化为list[],再存为tensor
                        '''

    # print("len(boxes)", len(boxes))
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        # 用cat进行拼接，维度为0，在指定维度上进行拼接---> (6,4)六行四列 和 (3,4)三行四列 cat（dim=0） 后变为 (7，4)七行四列

        boxes = torch.cat(boxes, 0)  # (n,4)

        # print("probs1", probs)
        probs = torch.cat(probs, 0)  # (n,)
        # print("probs2", probs)

        # print("cls_indexs", probs)
        cls_indexs = torch.cat(cls_indexs, 0)  # (n,)

    keep = nms(boxes, probs)
    print('keep:', keep)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.5):
    #  bboxes(tensor) [N,4]
    #  scores(tensor) [N,]

    # （1）计算预测边界框的面积

    # print(bboxes, scores)                         # 分别代表每个边框的信息，和置信度分数
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # 计算每个边框的面积

    _, order = scores.sort(0, descending=True)  # 对分数进行排序，留下角标信息， 降序
    # print('???:', order.unsqueeze(dim=1))

    keep = []
    # 查看order的值---验证查错可不看
    # print("order的值:", order)
    # print("order[0]的值:", order[0])
    # print("torch.tensor([order[0]])的值:", torch.tensor([order[0]]))
    # print("order增维:", torch.unsqueeze(order, dim=0).data)
    # print("order.item():", [order[0]].item())

    while order.numel() > 0:  # 看order里面有多少个数
        # 错误： ValueError: only one element tensors can be converted to Python scalars
        # 改动前：
        # i = order[0]
        # keep.append(i)

        # 改动后：-----> 先转化为列表，再将列表转为tensor形式

        # 反复循环避免报错 !!!!!!!!
        try:
            idx = order[0]
        except IndexError:
            break

        keep.append(idx)

        if order.numel() == 1:
            break
        # 在此用最大的置信度分数box的 x,y 坐标来作为衡量标准，用其他的框一次去作对比
        ''' clamp() 函数的作用是把一个值限制在一个上限和下限之间，当这个值超过最小值和最大值的范围时，在最小值和最大值之间选择一个值使用 '''
        xx1 = x1[order[1:]].clamp(min=x1[idx])  # 左下右上的xy坐标，相减得到W，h
        # print('x1[order[1:]]:', x1[order[1:]], '\n', 'x1[idx]:', x1[idx])
        # print('xx1:', xx1)

        yy1 = y1[order[1:]].clamp(min=y1[idx])
        xx2 = x2[order[1:]].clamp(max=x2[idx])
        yy2 = y2[order[1:]].clamp(max=y2[idx])

        w = (xx2 - xx1).clamp(min=0)  # 最小为零
        h = (yy2 - yy1).clamp(min=0)

        inter = w * h

        # 计算交并比
        ovr = inter / (areas[idx] + areas[order[1:]] - inter)
        # print('ovr:', ovr)

        #  UserWarning:This overload of nonzero is deprecated: nonzero()
        #  按照提示进行修改》》》》
        # 原本的：
        # ids = (ovr <= threshold).nonzero().squeeze()

        # 修改后的：
        ids = torch.nonzero((ovr <= threshold)).squeeze()  # 输出交并比小于阈值的脚标数值，如果没有就返回空值[]
        # print('ids:', ids)
        if ids.numel() == 0:
            break

        order = order[ids + 1]  # 相当于取出和之前最高置信度分数框交并比最大的框，在进行运算直到和其他框交并比小于阈值为止
        # print('修改后的order:', order)

    return torch.LongTensor(keep)


def imgshow(img):
    # img = img / 2 + 0.5
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(img)  # 显示图片
    plt.axis('on')  # 显示坐标轴
    plt.title('img')
    plt.show()


# start predict one image


def predict_gpu(model, image_name, root_path=''):
    result = []
    image = cv2.imread(root_path + image_name)

    h, w, _ = image.shape  # 读取图像的长和宽
    img = cv2.resize(image, (448, 448))  # 转化为448*448
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img)
    print('image.size:', img.size())

    # UserWarning: volatile was removed and now has no effect.Use`with torch.no_grad(): ` instead.
    # img = Variable(img[None, :, :, :],volatile=True)
    # 按照提示进行修改：
    # volatile：挥发性的-----------》 译为是否释放内存，如果volatile=True，相当于 with torch.no_grad():
    # 修改前：
    # img = Variable(img[None, :, :, :], volatile=True)
    # 修改后：
    with torch.no_grad():
        img = torch.autograd.Variable(img[None, :, :, :])

    img = img.cuda()
    pred = model(img)  # 1x7x7x30
    pred = pred.cpu()
    boxes, cls_indexs, probs = decoder(pred)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])
    return result


if __name__ == '__main__':
    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('../../weights/best.pth'))
    model.eval()
    model.cuda()
    # image_name = './data/example/000007.jpg'
    # image_name = './data/example/person.jpg'
    # image_name = './data/combine_doc/images/000007.jpg'
    image_name = './data/combine_doc/images/009889.jpg'

    # 用分隔符将图像路径划分开，只取最后的标题信息。
    title_name = image_name.split('/')[-1]
    print(title_name)

    image = cv2.imread(image_name)

    # 可查看图片
    # imgshow(image)

    print('predicting...')

    result = predict_gpu(model, image_name)

    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    # 根据自己的保存路径修改
    result_img = './data/example/' + "result_" + title_name
    # print(result_img)
    cv2.imwrite(result_img, image)
