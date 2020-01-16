import datetime
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

from Discriminator import Discriminator
from Generator import Generator


class FeatureVisualization():
    def __init__(self, img, selected_layer, model, model_name):
        """
        For GAN in Sketch Generation:
        Visualize output of convoluiton layer.
        :param img: a single image to pass the network. [num(1), channels, width, height]
        :param selected_layer: index of layer to show features
        :param model: network, named_children <main> should be iterable
        :param model_name: 'Generator' or 'Discriminator'
        """
        self.img = img
        self.selected_layer = selected_layer
        self.pretrained_model = model
        self.model_name = model_name

    def get_feature(self):
        """
        Get output feature of a layer.
        """
        img = self.img
        for index, layer in enumerate(self.pretrained_model.main):
            # print(index, layer)
            img = layer(img)
            if (index == self.selected_layer):
                return img

    def get_single_feature(self):
        feature = self.get_feature()
        # feature shapes [img_num(1 here), conv_kernel_num, output_size, output_size]
        feature = feature.view(feature.shape[1], feature.shape[0],
                               feature.shape[2], feature.shape[3])
        return feature

    def show_features(self, layer):
        """
        :param layer: index of model layer
        """
        feature = self.get_single_feature()
        # use sigmoid to normalize
        feature= 1.0/(1+torch.exp(-1*feature))
        feature_show(torchvision.utils.make_grid(feature), self.model_name, layer)


def normalize(x):
    """
    :param x: 2-D numpy array
    """
    x_min, x_max = x.min(0), x.max(0)
    x_norm = (x - x_min) / (x_max - x_min) # normalize
    return x_norm

def get_tsne(data, dimension):
    """
    :param data: 2-D numpy array
    :param dimension: 2 or 3
    """
    tsne = manifold.TSNE(n_components=dimension, init='pca', random_state=501)
    return tsne.fit_transform(data)

def show_distribution(real_norm, fake_norm, dimension, info, root='visualize/'):
    """
    For dataset in Sketch Generation:
    Plot the distribution of data.
    :param real_norm: normalized real images, size [n1, dimension]
    :param fake_norm: normalized fake images, size [n2, dimension]
    :param dimension: 2 or 3
    :param info: title of image
    :param root: root to save figures
    """
    fig = plt.figure()
    if dimension == 3:
        ax = Axes3D(fig)
        r = ax.scatter(real_norm[:, 0], real_norm[:, 1], real_norm[:, 2], color='b')
        f = ax.scatter(fake_norm[:, 0], fake_norm[:, 1], fake_norm[:, 2], marker='x', color='r')
    else:
        r = plt.scatter(real_norm[:, 0], real_norm[:, 1], color='b')
        f = plt.scatter(fake_norm[:, 0], fake_norm[:, 1], marker='x', color='r')
    plt.title(info)
    plt.axis([-200, 200, -200, 200])
    plt.legend([r, f], ['real', 'fake'], loc='best')
    plt.savefig(root+info+'.jpg', dpi=200)
    # plt.show()

def show_data_distribution(npoints=80, batch_size=64, dimension=2):
    """
    For dataset in Sketch Generation:
    Visualize the data distribution after trainning generator.
    :param npoints: number of sample real points
    :param batch_size: batch size of dataloader
    :param dimension: 2 or 3
    """
    transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataroot = 'SG_Dataset/'
    bird_dataset=datasets.ImageFolder(root=dataroot,transform=transform)
    dataloader = torch.utils.data.DataLoader(bird_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0)
    device = torch.device("cuda:0")

    real_all = np.zeros([batch_size, dimension])
    # Get all images in the dataset
    for i, data in tqdm(enumerate(dataloader)):
        real = data[0]
        real = real.reshape(batch_size, -1)
        real = real.numpy()
        real = get_tsne(normalize(real), dimension)
        if real_all.all() == 0:
            real_all = real
        else:
            try:
                real_all = np.concatenate((real_all, real), axis=0)
            except:
                pass
    
    # Get sample points to visualize
    rand_arr = np.arange(real_all.shape[0])
    np.random.shuffle(rand_arr)
    real_sample = real_all[rand_arr[:npoints]]

    # Get original fake points
    noise = torch.randn(batch_size, 100, 1, 1, device=device)
    netG = Generator(1).to(device)
    netG.apply(weights_init)
    fake = netG(noise)
    fake = fake.reshape(batch_size, -1)
    fake = fake.detach().cpu().numpy()
    fake = get_tsne(normalize(fake), dimension)

    show_distribution(real_sample, fake, 2, 'Original_distribution')

    # Get fake points after training
    for i in range(50, 550, 50):
        netG.load_state_dict(torch.load('/SketchGeneration/GAN_file/netG_epoch_'+str(i)+'.pth'))
        fake = netG(noise)
        fake = fake.reshape(64, -1)
        fake = fake.detach().cpu().numpy()
        fake = get_tsne(normalize(fake), 2)

        show_distribution(real_sample, fake, 2, 'Trained_distribution_of_epoch'+str(i))


def imshow(img):
    """
    Show a batch of image.
    :param img: Tensor [batchsize, channels, size, size]
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def feature_show(img, model_name, layer):
    """
    Show output of a activate layer.
    :param img: Tensor
    :param model_name: 'Generator' or 'Discriminator'
    :param layer: index of model layer
    """
    width = img.shape[2]
    height = img.shape[1]
    dpi = 200
    plt.figure(figsize=(width/dpi*3, height/dpi*3), dpi=dpi)
    npimg = img.detach().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    path = 'visualize/' + model_name + '_layer' + str(layer) + '.png'
    plt.savefig(path, dpi=dpi)
    plt.show()

def ConvVisualization(network, model_root):
    """
    Visualize a network's convolution layers.
    :param network: network, 'Generator' or 'Discriminator'
    :param model_root: .pth file, path to save parameters of model
    """
    if network == 'Generator':
        noise = torch.randn(1, 100, 1, 1, device=torch.device('cpu'))
        model = Generator(1)
        output_layer = [2, 5, 8, 11] # activate function layer
        model.load_state_dict(torch.load(model_root))
        for layer in output_layer:
            visitor = FeatureVisualization(noise, layer, model, network)
            visitor.show_features(layer)

    elif network == 'Discriminator':
        transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataroot = 'SG_Dataset/'
        dataset = datasets.ImageFolder(root=dataroot,transform=transform)
        feature_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=0)
        feature_iter = iter(feature_loader)
        image, _ = feature_iter.next()
        imshow(torchvision.utils.make_grid(image))
        model = Discriminator(1)
        output_layer = [1, 4, 7, 10] # activate function layer
        model.load_state_dict(torch.load(model_root))
        for layer in output_layer:
            visitor = FeatureVisualization(image, layer, model, network)
            visitor.show_features(layer)

    else:
        raise(Exception('model error')) 

if __name__ == "__main__":
    show_data_distribution()
    show_data_distribution(dimension=3)
    ConvVisualization('Discriminator', 'GAN_file/netD_epoch_250.pth')
    ConvVisualization('Generator', 'GAN_file/netG_epoch_250.pth')
    
    pass
