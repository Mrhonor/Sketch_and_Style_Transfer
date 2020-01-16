import torch
import torch.nn as nn
import torch.nn.functional as F
from __utils__ import weights_init_normal

from GeneratorResNet import GeneratorResNet
from Discriminator_n_layers import Discriminator_n_layers

def Create_nets(args):
    generator_AB = GeneratorResNet(args.input_nc_A,   args.input_nc_B ,args.n_residual_blocks)
    discriminator_B = Discriminator_n_layers(args.n_D_layers, args.input_nc_B)
    generator_BA = GeneratorResNet(args.input_nc_B,   args.input_nc_A ,args.n_residual_blocks)
    discriminator_A = Discriminator_n_layers(args.n_D_layers, args.input_nc_A)

    if torch.cuda.is_available():
        generator_AB = generator_AB.cuda()
        discriminator_B = discriminator_B.cuda()
        generator_BA = generator_BA.cuda()
        discriminator_A = discriminator_A.cuda()

    if args.epoch_start != 0:
        # Load pretrained models
        generator_AB.load_state_dict(torch.load('saved_models/%s/G__AB_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator_B.load_state_dict(torch.load('saved_models/%s/D__B_%d.pth' % (opt.dataset_name, opt.epoch)))
        generator_BA.load_state_dict(torch.load('saved_models/%s/G__BA_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator_A.load_state_dict(torch.load('saved_models/%s/D__A_%d.pth' % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator_AB.apply(weights_init_normal)
        discriminator_B.apply(weights_init_normal)
        generator_BA.apply(weights_init_normal)
        discriminator_A.apply(weights_init_normal)

    return generator_AB, discriminator_B, generator_BA, discriminator_A