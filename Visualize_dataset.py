import SDoG
import os
from data_loader import CUB
import matplotlib.pyplot as plt
import random
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# 数据集1 CUB-200-2011
# 本来是用于图像分类的数据集，经过处理变成GAN的训练集
# 原图像->提取bounding-box->切割图像->SDoG算子提取边缘生成简笔画

cub200_root = '../cub200/'
origin_cub_train = CUB(cub200_root)
origin_cub_test = CUB(cub200_root, train=None)
sketch_cub_train = CUB(cub200_root, transform=SDoG.XDOG)
sketch_cub_test = CUB(cub200_root, train=None, transform=SDoG.XDOG)

len_dataset = len(origin_cub_train)+len(origin_cub_test)
print('The size of CUB-200 dataset: ', len_dataset)

choice = random.randint(0, len(origin_cub_train))
plt.axis('off')
origin, _ = origin_cub_train._get_origin_image(choice)
plt.imshow(origin)
plt.axis('off')
plt.imshow(origin_cub_train._show_bounding_box(str(choice+1)))
image, _ = origin_cub_train[choice]
plt.axis('off')
plt.imshow(image)
sketch, _ = sketch_cub_train[choice]
plt.axis('off')
plt.imshow(sketch)