import heapq
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as utils
from PIL import Image

from Discriminator import Discriminator
from Generator import Generator

outf = '../GAN_file/'

nz=100
device = torch.device("cpu")


manualSeed = random.randint(1, 10000)
print("Random Seed: ",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True

netG = Generator(0).to(device)
netG.load_state_dict(torch.load(outf+'netG_epoch_250.pth',map_location='cpu'))

netD=Discriminator(0).to(device)
netD.load_state_dict(torch.load(outf+'netD_epoch_250.pth',map_location='cpu'))

#测试时的输入图像
input_path= 'img0.jpg'
input=Image.open(input_path)
# input_sketch =input.crop((0,0,64,64))
input_sketch = input.resize((64,64))
input_sketch_gray=input_sketch.convert('L')
input_tensor=transforms.ToTensor()(input_sketch)
input_tensor=(input_tensor-0.5)*2
input_sketch_gray_hist=(torch.tensor(input_sketch_gray.histogram(),dtype=float)+1)/4352

#生成随机噪声，然后通过生成器生成图片
noise = torch.randn(64, nz, 1, 1, device=device)
noise_img=netG(noise)
criterion = nn.KLDivLoss(size_average=False)
criterion1=nn.BCELoss()


loss=[]
for i in range(64):
    path='./fake/fake_samples_'+str(i)+'.jpg'
    utils.save_image(noise_img[i].detach(),path,
                     normalize=True)
    img=Image.open(path)
    img_sketch=img.crop((0,0,64,64))
    img_sketch_gray=img_sketch.convert('L')
    img_sketch_gray_hist=(torch.tensor(img_sketch_gray.histogram(),dtype=float)+1)/4352
    log_img_sketch_gray_hist=torch.log(img_sketch_gray_hist)
    lo=criterion(log_img_sketch_gray_hist,input_sketch_gray_hist)
    loss.append(lo)

re1 = map(loss.index, heapq.nsmallest(5, loss)) #求loss中最小的5个索引
re2 = heapq.nsmallest(5, loss) #求最小的5个KL损失
re1=list(re1)
print(re1)
print(re2)

index=loss.index(min(loss))
print(loss)
print(index)


def calc_vector2(img):
   two_value_img = img.convert('1')
   x = np.zeros(two_value_img.size[0]*2)
   num = 0
   for i in range(two_value_img.size[0]):
      sum = 0
      sum2 = 0
      for j in range(i):
         if(two_value_img.getpixel((j,i-j)) == 255):
            sum = sum + 1
            num = num + 1
         if(two_value_img.getpixel((two_value_img.size[0]-j-1, two_value_img.size[0]+j-i-1)) == 255):
            sum2 = sum2 + 1
            num = num + 1
      x[i] = sum
      x[127-i] = sum2
   return x
x1 = calc_vector2(input_sketch_gray)

cri=nn.MSELoss()
loss_1=[]
for i in range(5):
    path = './fake/fake_samples_' + str(re1[i]) + '.jpg'
    img = Image.open(path)
    img_sketch = img.crop((0, 0, 64, 64))
    img_sketch_gray = img_sketch.convert('L')
    x2=calc_vector2(img_sketch_gray)
    loss_2=cri(torch.tensor(x1),torch.tensor(x2))
    loss_1.append(loss_2)

min_index = loss_1.index(min(loss_1))
print(re1[min_index])
print(loss_1)

#初始化z向量
z=noise[re1[min_index]]
z.requires_grad=True
optimizer=optim.Adam([z],lr=0.01)

criterion2=nn.L1Loss()

for i in range(500):
    path='./fake_change/fake_samples_' + str(i) + '.jpg'
    optimizer.zero_grad()
    output = netG(z.unsqueeze(0))
    output1 = output.squeeze(0)
    utils.save_image(output1.detach(), path,
                    normalize=True)
    p=output1[:,:,:64]
    contextual_loss=criterion2(p,input_tensor)
    dis = netD(output)
    label = torch.full((1,), 0, device=device)
    perceptual_loss = criterion1(dis, label)
    whole_loss =contextual_loss-0.21* perceptual_loss
    whole_loss.backward()
    optimizer.step()



noise[re1[min_index]]=z
noise_img_final=netG(noise)
utils.save_image(noise_img_final[re1[min_index]].detach(),
                    'final1_img.jpg',
                    normalize=True)

img_1=Image.open('final1_img.jpg')
array1=np.array(input_sketch)
array2=np.array(img_1)
array2=array2[:,64:,:]
img_final = np.concatenate((array1,array2), 1)
img_final = Image.fromarray(img_final)
img_final.save('img_final.jpg')
