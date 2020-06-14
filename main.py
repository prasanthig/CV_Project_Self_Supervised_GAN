"""
    Pytorch implementation of Self-Supervised GAN
    Reference: "Self-Supervised GANs via Auxiliary Rotation Loss"
    Authors: Ting Chen,
                Xiaohua Zhai,
                Marvin Ritter,
                Mario Lucic and
                Neil Houlsby
    https://arxiv.org/abs/1811.11212 CVPR 2019.
    Script Author: Vandit Jain. Github:vandit15
"""
import torch
import torch.optim as optim
from dataloaders import get_STL10_dataloaders
from model_new import Generator, Discriminator
from training import Trainer
import random
import sys
from torchsummary import summary

img_size = (48, 48, 3)
batch_size = 64
#Hyper Paramenters
g_lr = 1e-4
d_lr = 4e-4
betas = (0., .99)

data_loader, _, _ = get_STL10_dataloaders(batch_size=batch_size)

generator = Generator(z_size = 128, channel = 3, output_size=48)
discriminator = Discriminator(channel = 3, ssup = True)

# Initialize optimizers
G_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=betas)


# Train model
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  weight_rotation_loss_d = 1.0, weight_rotation_loss_g = 0.2, critic_iterations=1,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

