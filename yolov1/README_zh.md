写代码的原博主采用的是pytorch2.0的框架写的这个yolo1，里面有很多的函数在pytorch3.0中已经不适用了。

代码部分有些地方感觉不那么精简，但是只是用来学习YOLO1的话还是没有什么问题的，足够看了，
当时github上面找代码的时候，找了几个感觉这个原博主写的还比较全，就是用的这个来进行学习的。

按照错误的提示，修改了里面的部分内容，最后按照自己喜欢的格式整合了一下。

如果使用的是pytorch3.0以上的版本，下载后基本上只需要修改文件路径就行了，因为在代码中为了容易区分，
我有些地方用的是相对路径，有些地方用的是绝对路径，如果相对路径运行着有问题，就改成绝对路径吧。

建议: ①先看看论文，对yolo有个大致的了解
     ②结合论文看代码中yolo网络模型的构成，LOSS损失函数 
     ③进行训练看最后的结果

代码目录树结构如下:
     --yolov1(根目录)
          --data(数据集目录)
               --combine_doc
                    --images(数据集可自行制作)
                    --labels
                    ...
               --example
               --log.txt
          --model(模型定义目录)
          --result(测试结果目录)
          --run(训练,测试等运行目录)
          --weights(保存权重目录)
          --dataset.py(定义数据集)
          --train.py(训练脚本)
          ...(论文和readme)



# 训练自己的数据集

**2023.10.31**
   1.