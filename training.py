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
import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 weight_rotation_loss_d, weight_rotation_loss_g, critic_iterations=2, print_every=50, use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': []}
        self.accuracy_list= {'RA':[], 'FA':[]}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.weight_rotation_loss_d = weight_rotation_loss_d
        self.weight_rotation_loss_g = weight_rotation_loss_g
        self.num_rotation = 4 #4 rotation angles

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def loss_hinge_dis(self, d_real_logits, d_fake_logits):
        loss = torch.mean(F.relu(1.0 - d_real_logits))
        loss += torch.mean(F.relu(1.0 + d_fake_logits))
        return loss

    def loss_hinge_gen(self, d_fake_logits):
        loss = - torch.mean(d_fake_logits)
        return loss

    def accuracy(self, pred, y):
        pred_ind = pred.max(1, keepdim=True)[1]
        accuracy = torch.mean(pred_ind.eq(y.view_as(pred_ind)).cpu().float())
        return accuracy

    def _discriminator_train_iteration(self, data, generated_data, batch_size):
        """ """
        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        _, d_real_pro_logits, d_real_rot_logits, d_real_rot_prob = self.D(data)
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(generated_data)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = self.loss_hinge_dis(d_real_pro_logits, g_fake_pro_logits)

        # Add auxiiary rotation loss
        rot_labels = torch.zeros(4*batch_size, dtype = torch.int64).cuda()
        for i in range(4*batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2*batch_size:
                rot_labels[i] = 1
            elif i < 3*batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3

        rot_labels_one_hot = F.one_hot(rot_labels.to(torch.int64), 4).float()
        d_real_class_loss = torch.mean(F.binary_cross_entropy_with_logits(input = d_real_rot_logits, target = rot_labels_one_hot))

        d_loss += self.weight_rotation_loss_d * d_real_class_loss
        d_loss.backward(retain_graph=True)

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.data)

        real_accuracy = self.accuracy(d_real_rot_logits, rot_labels)

        # Record real accuracy
        self.accuracy_list['RA'].append(real_accuracy)

    def _generator_train_iteration(self, generated_data, batch_size):
        """ """
        self.G_opt.zero_grad()

        # Calculate loss and optimize
        _, g_fake_pro_logits, g_fake_rot_logits, g_fake_rot_prob = self.D(generated_data)
        #g_loss = - torch.sum(g_fake_pro_logits)
        g_loss = self.loss_hinge_gen(g_fake_pro_logits)
        # add auxiliary rotation loss
        rot_labels = torch.zeros(4*batch_size,dtype = torch.int64).cuda()
        for i in range(4*batch_size):
            if i < batch_size:
                rot_labels[i] = 0
            elif i < 2*batch_size:
                rot_labels[i] = 1
            elif i < 3*batch_size:
                rot_labels[i] = 2
            else:
                rot_labels[i] = 3

        rot_labels_one_hot = F.one_hot(rot_labels.to(torch.int64), 4).float()
        g_fake_class_loss = torch.mean(F.binary_cross_entropy_with_logits(
            input = g_fake_rot_logits,
            target = rot_labels_one_hot))


        g_loss += self.weight_rotation_loss_g * g_fake_class_loss

        g_loss.backward(retain_graph=True)
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.data)
        fake_accuracy = self.accuracy(g_fake_rot_logits, rot_labels)
        # Record fake accuracy
        self.accuracy_list['FA'].append(fake_accuracy)

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            # Get generated data
            data = data[0]
            batch_size = data.size()[0]
            generated_data = self.sample_generator(batch_size)

            x = generated_data
            x_90 = x.transpose(2,3)
            x_180 = x.flip(2,3)
            x_270 = x.transpose(2,3).flip(2,3)
            generated_data = torch.cat((x, x_90, x_180, x_270),0)

            x = data
            x_90 = x.transpose(2,3)
            x_180 = x.flip(2,3)
            x_270 = x.transpose(2,3).flip(2,3)
            data = torch.cat((x,x_90,x_180,x_270),0)

            self.num_steps += 1
            self._discriminator_train_iteration(data, generated_data, batch_size)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(generated_data, batch_size)

            writer.add_scalar('Discriminator Loss',self.losses['D'][-1],self.num_steps)
            writer.add_scalar('Real Accuracy',self.accuracy_list['RA'][-1],self.num_steps)

            if self.num_steps > self.critic_iterations:
                writer.add_scalar("Generator Loss",self.losses['G'][-1], self.num_steps)
                writer.add_scalar('Fake Accuracy',self.accuracy_list['FA'][-1],self.num_steps)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("Real Accuracy: {}".format(self.accuracy_list['RA'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))
                    print("Fake Accuracy: {}".format(self.accuracy_list['FA'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        print(self.D)
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            if epoch%5 == 0:
                torch.save({
                            'gen_state_dict': self.G.state_dict(),
                            'dis_state_dict': self.D.state_dict(),
                            'gen_optimizer_state_dict': self.G_opt.state_dict(),
                            'dis_optimizer_state_dict': self.D_opt.state_dict(),
                            'epoch' : epoch,
                            'steps': self.num_steps,
                            'discriminator_loss': self.losses['D'],
                            'generator_loss': self.losses['G'],
                            'real_accuracy': self.accuracy_list['RA'],
                            'fake_accuracy': self.accuracy_list['FA']
                            }, './SSGANModel/ssgan_' + str(epoch) + '.pt')


            if save_training_gif:
                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose((img_grid.numpy()), (1, 2, 0))
                img_grid = 255*(img_grid - np.min(img_grid))/np.ptp(img_grid)
                # Add image grid to training progress
                training_progress_images.append(img_grid.astype(np.uint8))
                imageio.imwrite('./Gen_Images/training_{}.png'.format(epoch),img_grid)
                imageio.mimsave('./Gen_Gifs/training_{}_epochs.gif'.format(epoch), training_progress_images)

        if save_training_gif:
            imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
                            training_progress_images)
        writer.close()

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]