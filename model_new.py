import os.path
import tarfile, sys, math
from six.moves import urllib
from ops import Residual_G, Residual_D, snlinear,snconv2d, Self_Attn
import numpy as np
import scipy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.autograd import grad as torch_grad
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

class Discriminator(nn.Module):
    def __init__(self, ssup, channel, featOnly=False):
        super(Discriminator, self).__init__()
        self.ssup = ssup
        self.featOnly = featOnly
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.self_attn = Self_Attn(in_channels = 64)
        self.re1 = Residual_D(channel, 64, down_sampling = True, is_start = True)
        self.re2 = Residual_D(64, 128, down_sampling = True)
        self.re3 = Residual_D(128, 256, down_sampling = True)
        self.re4 = Residual_D(256, 512, down_sampling = True)
        self.re5 = Residual_D(512, 1024)

        self.fully_connect_gan2 = snlinear(1024, 1)
        self.fully_connect_rot2 = snlinear(1024, 4)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)


    def forward(self, x):
        re1 = self.re1(x)           #24x24
        a0 = self.self_attn(re1)    #24x24
        re2 = self.re2(a0)          #12x12
        re3 = self.re3(re2)         #6x6
        re4 = self.re4(re3)         #3x3
        re5 = self.re5(re4)         #1x1
        re5 = self.relu(re5)
        re5 = torch.sum(re5,dim = (2,3))

       
        if self.featOnly:
            return re5
       
        gan_logits = self.fully_connect_gan2(re5)
        if self.ssup:
            rot_logits = self.fully_connect_rot2(re5)
            rot_prob = self.softmax(rot_logits)

        if self.ssup:
            return self.sigmoid(gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.sigmoid(gan_logits), gan_logits

class Generator(nn.Module):
    def __init__(self, z_size, channel, output_size = 48):
        super(Generator, self).__init__()
        self.output_size = output_size
        s = 4
        if self.output_size == 48:
            s = 6
        self.s = s
        self.z_size = z_size
        self.fully_connect = snlinear(z_size, s*s*512)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv_res4 = snconv2d(64,channel, padding = 1, kernel_size = 3, stride = 1)
        self.self_attn = Self_Attn(in_channels = 256)

        self.re1 = Residual_G(512, 256, up_sampling = True)
        self.re2 = Residual_G(256, 128, up_sampling = True)
        self.re3 = Residual_G(128, 64, up_sampling = True)
        self.bn = nn.BatchNorm2d(64)
        self.apply(init_weights)

    def forward(self, x):
        d1 = self.fully_connect(x)
        d1 = d1.view(-1, 512, self.s, self.s)
        d2 = self.re1(d1) #12x12
        d3 = self.self_attn(d2) #24x24
        d4 = self.re2(d3) #24x24
        
        d4 = self.re3(d4) #48x48
        d4 = self.relu(self.bn(d4))
        d5 = self.conv_res4(d4) #48x48
        return self.tanh(d5)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.z_size))

