# 简笔画生成图像+风格迁移

2019/12/31 更新：

更新readme.md，更新数据集使用说明

更新项目结构，Generator和Discriminator单独放在一个py文件，weights_init函数放在\_\_utils\_\_.py里方便调用

更新Visualize.py，对于GAN部分，完成Generator生成图像分布的可视化，主要使用t-SNE降维方法降到2维或3维展示；完成Generator和Discriminator的卷积可视化（Generator的卷积可视化没有什么可解释性，因为输入是随机噪声而且卷积输出不是三通道的，只能观察到像素级的变化；从Discriminator的卷积可视化结果看出其分类效果还不是很好，没有提取到很多有效的特征，可能要继续加大Epoch，或者先用噪声和real_data预训练Discriminator一会儿再和Generator一起训练？）

在GAN.py中完成Loss变化曲线的可视化，数据结果和图像结果默认保存在visualize文件夹中

在Visualize_dataset.ipynb中简单复现了部分鸟类数据集的处理过程，但是做到一半发现这个好像没有什么用。。如果后面写论文需要再继续做



TODO：

风格迁移部分可视化

（GAN运行时KL散度和JS散度变化的可视化）



## 数据载入用法
可由以下链接获得经过人工筛选的数据集，放在项目的同级文件夹dataset中即可。

链接：https://pan.baidu.com/s/1Vzov0thOXT_-f2RWkmWgMA  提取码：u841 

以下是由原始数据集生成本项目所需数据集的步骤：



**data_loader.py**

原始数据集：

 CUB_200_2011数据集下载地址 http://www.vision.caltech.edu/visipedia/CUB-200-2011.html 

1. 数据集放于该项目的上一个目录
2. CUB_200类调用用法
CUB_200类是处理鸟内数据集的方法
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
3. Car类调用方法

Car类是处理斯坦福大学的Car dataset数据集的类

Car_dataset数据集下载地址: [官网](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 

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

4. Concatenate函数的用法

该函数用于将简笔画图像和原图拼接在一起，训练时数据集使用该函数拼接生成的图像
~~~
def Concatenate(root, path1, path2):
~~~
**root**:数据集存放目录

**path1**:简笔画图像所在文件夹

**path2**:原图所在文件夹

拼接后生成的图像位于 **root/dataset/bird**\





----
## 运行程序
目前默认读入鸟类数据集
~~~
python data_loader.py
python GAN.py
~~~

注意：运行GAN.py和Visualize.py时，需分别在项目中创建GAN_file文件夹和visualize文件夹保存结果。