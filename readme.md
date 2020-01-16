# 简笔画生成图像+风格迁移

## 课题背景

(之前开题报告的时候好像没讲这一部分，按那个PPT的思路就行了)

## 相关工作

(主要介绍一下那篇Contextual GAN的论文)

## 数据集准备
可由以下链接获得经处理的数据集，将SG_dataset和ST_dataset两个文件夹放在项目的同级目录中即可。

链接：https://pan.baidu.com/s/1yLY2L0bKefEcqOMJq4J36w 提取码：wsl7 



以下是由原始数据集生成本项目所需数据集的步骤：

Sketch Generation部分：

[1]. 获取原始数据集CUB_200_2011，[下载地址]( http://www.vision.caltech.edu/visipedia/CUB-200-2011.html )  将CUB_200文件夹放在项目的同级目录中

[2]. 获取原始数据集Car_dataset。[下载地址](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 同样将Car文件夹放在项目的同级目录中

**[3]. 在SketchGeneration文件夹中运行data_loader.py**

```
cd SketchGeneration
python data_loader.py
```

详解：

CUB_200_2011:

~~~
class CUB_200(Dataset):
    def __init__(self, root, train=True, transform=None):
~~~
**root**: 数据集存放目录，默认为"../"

**train**: 获取训练数据集还是测试数据集

**transform**: 对数据集进行的特殊操作，参数应传入一个参数为图像的函数，如使用XDOG处理数据时传入:
~~~
transform=SDoG.XDOG
~~~
获得处理后的图像:
~~~
cub = CUB_200("../", transform=SDoG.XDOG)
for img, label in cub:
~~~
Car_dataset:

~~~
class Car(Dataset):
    def __init__(self, root, train=True, transform=None):
~~~
大体信息同上，需要额外注意的为获取bounding boxes数据时，要注意修改文件的相对位置。
~~~~
self.bounding_boxes_file = os.path.join(root, 'car_devkit/devkit/cars_train_annos.mat')
~~~~
另外需要注意的是，该类获得图片的调用方法略有不同
~~~~
car = Car('../', transform=SDoG.XDOG)
img = car.getitem(image_id)
~~~~

Concatenate函数的用法

该函数用于将简笔画图像和原图拼接在一起，训练时数据集使用该函数拼接生成的图像
~~~
def Concatenate(root, path1, path2):
~~~
**root**:数据集存放目录

**path1**:简笔画图像所在文件夹

**path2**:原图所在文件夹

拼接后生成的图像位于 **root/SG_Dataset/bird**/ 或**root/SG_Dataset/car/**



至此完成Sketch Generation部分数据集准备。（实际使用的数据集经过人工筛选处理）



Style Transfer部分:

此部分没有对原始数据集进行额外处理，可由此获得原始数据集：[下载地址](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) (使用了maps文件夹中的图像，将maps文件夹放在ST_Dataset文件夹中即可)



----
## 运行程序
目前Sketch Generation部分默认使用Car_dataset
~~~
cd SketchGeneration
python GAN.py
~~~

运行完GAN.py得到模型参数后，在目录下放入img0.jpg。运行

```
python implementation.py
```

可以得到生成出来的图片img_final.jpg。运行时间大约为1分钟。

```
cd ../StyleTransfer
python cycleGAN.py
```



## 方法介绍

1、数据集的使用处理

2、Sketch Generation部分（训练完成后的噪声处理）

3、可视化部分(T-SNE降维展示)

4、Style Transfer部分（Cycle GAN和CNN的结果对比）



## 实验结果和分析



## Contributor



## References

