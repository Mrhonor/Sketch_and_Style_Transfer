import torch.nn as nn
import random
import  torch
import math
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.optim as optim
from PIL import Image
import heapq
import numpy as np
from scipy import stats

nz=100
N=10
nc=3
ngf=64
ndf=64
device = torch.device("cpu")


manualSeed = 3279
print("Random Seed: ",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, (4,8), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1,bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 128
        )
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 8
            nn.Conv2d(ndf * 8, 1, (4,8), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netG = Generator(0).to(device)
netG.load_state_dict(torch.load('netG_epoch_250.pth',map_location='cpu'))

netD=Discriminator(0).to(device)
netD.load_state_dict(torch.load('netD_epoch_250.pth',map_location='cpu'))


noise = torch.randn(64, nz, 1, 1, device=device)
noise_img=netG(noise)
criterion = nn.KLDivLoss(size_average=False)
criterion1=nn.BCELoss()

#测试时的输入图像
input=Image.open('img0.jpg')
input_sketch =input.crop((0,0,64,64))
input_sketch_gray=input_sketch.convert('L')
input_sketch_gray_hist=(torch.tensor(input_sketch_gray.histogram(),dtype=float)+1)/4352


path1='./test.jpg'
iii=noise.clone().detach()
r1=netG(iii[31:33])
ttt=noise_img[31]
k1=np.array(r1[0].detach())
k2=np.array(ttt.detach())
lossasdasd = np.linalg.norm(k1-k2)
print(lossasdasd)
print(r1.size())
#r2 = r1.squeeze(0)
utils.save_image(r1[0].detach(), path1,
                         normalize=True)
r3_img = Image.open(path1)
r4_sketch = r3_img.crop((0, 0, 64, 64))
r5_sketch_gray = r4_sketch.convert('L')
r6sketch_gray_hist = (torch.tensor(r5_sketch_gray.histogram(), dtype=float) + 1) / 4352
log_out_img_sketch_gray_hist = torch.log(r6sketch_gray_hist)
contextual_loss=criterion(log_out_img_sketch_gray_hist,input_sketch_gray_hist)
print('asd')
print(contextual_loss)




loss=[]
for i in range(64):
    path='./fake/fake_samples_'+str(i)+'.jpg'
    #noise1 =noise[i].clone().detach()
    #noise_img1=netG(noise1.unsqueeze(0))
    utils.save_image(noise_img[i].detach(),path,
                     normalize=True)
    img=Image.open(path)
    img_sketch=img.crop((0,0,64,64))
    img_sketch_gray=img_sketch.convert('L')
    img_sketch_gray_hist=(torch.tensor(img_sketch_gray.histogram(),dtype=float)+1)/4352
    log_img_sketch_gray_hist=torch.log(img_sketch_gray_hist)
    lo=criterion(log_img_sketch_gray_hist,input_sketch_gray_hist)
    loss.append(lo)

re1 = map(loss.index, heapq.nsmallest(5, loss)) #求最小的5个索引
re2 = heapq.nsmallest(5, loss) #求最小的5个元素
re1=list(re1)
print(re1)
print(re2)

index=loss.index(min(loss))
index1=loss.index(max(loss))
print(loss)
print(index1)
print(index)

def calc_vector2(img):
   two_value_img = img.convert('1')
   x = np.zeros(two_value_img.size[0])
   num = 0
   for i in range(two_value_img.size[0]):
      sum = 0
      for j in range(i):
         if(two_value_img.getpixel((j,i-j)) == 0):
            sum = sum + 1
            num = num + 1
      x[i] = sum
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


z=noise[re1[min_index]]
z.requires_grad=True
optimizer=optim.Adam([z],lr=0.001)

for i in range(200):
    path='./fake_change/fake_samples_' + str(i) + '.jpg'
    optimizer.zero_grad()
    output = netG(z.unsqueeze(0))
    output1 = output.squeeze(0)
    utils.save_image(output1.detach(), path,
                     normalize=True)
    out_img = Image.open(path)
    out_img_sketch = out_img.crop((0, 0, 64, 64))
    out_img_sketch_gray = out_img_sketch.convert('L')
    out_img_sketch_gray_hist = (torch.tensor(out_img_sketch_gray.histogram(), dtype=float) + 1) / 4352
    out_img_sketch_gray_hist.requires_grad=True
    log_out_img_sketch_gray_hist = torch.log(out_img_sketch_gray_hist)
    contextual_loss=criterion(log_out_img_sketch_gray_hist,input_sketch_gray_hist)
    print('\n')
    print(contextual_loss)
    print(contextual_loss.requires_grad)
    dis = netD(output)
    label = torch.full((1,), 0, device=device)
    perceptual_loss = criterion1(dis, label)
    whole_loss = contextual_loss--0.01* perceptual_loss
    print(whole_loss.dtype)
    whole_loss.backward()
    print(z.grad)
    optimizer.step()


img_final=netG(z.unsqueeze(0))
utils.save_image(img_final.detach(),
                    'final_img.jpg',
                    normalize=True)
#print(output.size())
#output1=torch.mean(output)
#output1.backward()
#print(z.grad)

#img1=transforms.ToPILImage()(img[0])
#img1.show()




