# 简笔画生成图像+风格迁移
## 数据载入用法
**data_loader.py**
1. 数据集放于该项目的上一个目录
2. CUB_200类调用用法
CUB_200类是处理鸟内数据集的方法

CUB_200_2011数据集下载地址 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>
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

Car_dataset数据集下载地址:
[官网](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 
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

拼接后生成的图像位于 **root/dataset/bird**

----
## 运行程序
目前默认读入鸟类数据集
~~~
python data_loader.py
python GAN.py
~~~