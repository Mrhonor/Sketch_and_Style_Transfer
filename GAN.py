import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

from __utils__ import weights_init
from Discriminator import Discriminator
from Generator import Generator

nz=100
nc=3
ngf=64
ndf=64
niter=501
lr=0.0002
beta1=0.5
ngpu=1 #2
netG=''
netD=''
outf='GAN_file/'
batchSize=64
device = torch.device("cuda:0")
try:
    os.makedirs(outf)
except OSError:
    pass

manualSeed = random.randint(1, 10000)
print("Random Seed: ",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
cudnn.benchmark = True

transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataroot='../dataset/'
bird_dataset=datasets.ImageFolder(root=dataroot,transform=transform)
dataloader = torch.utils.data.DataLoader(bird_dataset, batch_size=batchSize,
                                         shuffle=True,num_workers=0)


netG = Generator(ngpu).to(device)
netG.apply(weights_init)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)


criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

Loss_D = []
Loss_G = []

for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
#         Loss_D.append(errD.item())
#         Loss_G.append(errG.item())
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            utils.save_image(real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(fixed_noise)
            utils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

    Loss_D.append(errD.item())
    Loss_G.append(errG.item())

    # do checkpointing
    if epoch%50==0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))


Loss_D = ['{:.4f}'.format(i) for i in Loss_D] 
Loss_G = ['{:.4f}'.format(i) for i in Loss_G] 
with open('visualize/Loss.txt','wt') as f:
    print(Loss_D, file=f)
    print('\n', file=f)
    print(Loss_G, file=f)

D, = plt.plot(Loss_D, color='r', label='Loss_D')
G, = plt.plot(Loss_G, color='b', label='Loss_G')
plt.legend()
plt.savefig('visualize/GAN_Loss.png', dpi=200)
# plt.show()
