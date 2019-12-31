import torch.nn as nn
import random
import  torch
import math
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.optim as optim
from PIL import Image
import numpy as np

nz=100
N=10
nc=3
ngf=64
ndf=64
device = torch.device("cpu")


manualSeed = random.randint(1, 10000)
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
netG.load_state_dict(torch.load('netG_epoch_200.pth',map_location='cpu'))

netD=Discriminator(0).to(device)
netD.load_state_dict(torch.load('netD_epoch_200.pth',map_location='cpu'))

array= np.random.randn(3,64,128)
array[:,:,:64]=1
array[:,:,64:]=0
mask=torch.BoolTensor(array,device=device)

noise = torch.randn(10, nz, 1, 1, device=device)
noise_img=netG(noise)
criterion = nn.KLDivLoss()
criterion1=nn.BCELoss()

input=Image.open('img0.jpg')
input_tensor=transforms.ToTensor()(input)
input_tensor.requires_grad=True
input_tensor_sketch=torch.masked_select(input_tensor,mask)
print(input_tensor_sketch.size())
print(input_tensor_sketch.requires_grad)

loss=[]
for i in range(10):
    kk=noise_img[i]
    kk1=torch.masked_select(kk,mask)
    lo=criterion(input_tensor_sketch,kk1)
    loss.append(lo.detach())


index=loss.index(min(loss))
z=noise[index]
#z=z.unsqueeze(0)
z.requires_grad=True



#optimizer=optim.SGD([z], lr = 0.01, momentum=0.9)

print(z.grad)
output=netG(z.unsqueeze(0))
output_sketch=torch.masked_select(output.squeeze(0),mask)
iausd=torch.randn(3,64,128,device=device,requires_grad=True)
iausd1=torch.masked_select(iausd,mask)
contextual_loss=criterion(iausd1,output_sketch)
contextual_loss.backward()


#print(output.size())
#output1=torch.mean(output)
#output1.backward()
print(z.grad)

#print(z.grad)



#img=netG(z)
#img1=transforms.ToPILImage()(img[0])
#img1.show()
#utils.save_image(img.detach(),
 #                   'fake_samples_epochllll.png',
  #                  normalize=True)




