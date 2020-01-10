from PIL import Image, ImageEnhance
from pylab import *
import numpy as np
from scipy.ndimage import filters
import glob, os

# out_dir = 'out'
# if not os.path.exists(out_dir): os.mkdir(out_dir)

def XDOG(im):
    Gamma = 0.89 #0.97  ##过滤线条
    Phi = 200
    Epsilon = 0.1
    k = 2.5 #2.5  ##过滤线条
    Sigma = 2 #1.5 ##调整细粒度

    im = im.convert('L')
    im = np.array(ImageEnhance.Sharpness(im).enhance(3.0))
    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma* k)
    differencedIm2 = im2 - (Gamma * im3)
    (x, y) = np.shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + np.tanh(Phi * (differencedIm2[i, j]))


    gray_pic=differencedIm2.astype(np.uint8)
    final_img = Image.fromarray( gray_pic)
    # final_img.save(os.path.join(out_dir, filename))
    return final_img