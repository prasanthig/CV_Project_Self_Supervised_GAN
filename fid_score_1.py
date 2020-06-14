#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from model_new import Generator
from torch.autograd import Variable
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

num_images = 100000
batch_size = 100


def get_activations(files, model, batch_size=100, dims=2048, cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
    Make sure that the number of samples is a multiple of
    the batch size, otherwise some samples are ignored. This
    behavior is retained to match the original FID score
    implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
    of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
    activations of the given tensor when feeding inception with the
    query tensor.
    """
    model.cuda()
    model.eval()

    n_batches = num_images // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        images = files[i]
        start = i * batch_size
        end = start + batch_size
        pred = model(images.cuda())[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
    inception net (like returned by the function 'get_predictions')
    for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
    representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
    representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    #print("MU1: ",mu1," MU2: " ,mu2," SIGMA1: ", sigma1," SIGMA 2 ", sigma2)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
        'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=100, dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
    batch size batch_size. A reasonable batch size
    depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
    number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
    the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
    the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(images, model, batch_size, dims, cuda):

    m, s = calculate_activation_statistics(images, model, batch_size, dims, cuda)

    return m, s


def calculate_fid_given_paths(real_images, fake_images, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    """for p in paths:
    if not os.path.exists(p):
        raise RuntimeError('Invalid path: %s' % p)"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

        m1, s1 = _compute_statistics_of_path(real_images, model, batch_size,
        dims, cuda)
        m2, s2 = _compute_statistics_of_path(fake_images, model, batch_size,
        dims, cuda)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value


if __name__ == '__main__':  
    dims = 2048
    steps_per_epoch = 1563
    fid_scores = []
    steps = [20 * steps_per_epoch, 40 * steps_per_epoch,60 * steps_per_epoch,80 * steps_per_epoch,100 * steps_per_epoch,
            120 * steps_per_epoch,140 * steps_per_epoch,150 * steps_per_epoch]

    print(steps,len(steps))
    all_transforms = transforms.Compose([
                                    transforms.Resize(48),
                                    transforms.CenterCrop(48),
                                    transforms.ToTensor()
                                    ])

    #100000 images wiht 64 batch size in training => 100000/64 = 1563 steps per epoch
    
    model_checkpoints = ['./SSGANModel/ssgan_20.pt', './SSGANModel/ssgan_40.pt', './SSGANModel/ssgan_60.pt',
                         './SSGANModel/ssgan_80.pt', './SSGANModel/ssgan_100.pt',
                         './SSGANModel/ssgan_120.pt','./SSGANModel/ssgan_140.pt',
                         './SSGANModel/ssgan_150.pt']

    train_data = datasets.STL10('../stl10_data', split='unlabeled', download=False, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    generator = Generator(z_size = 128, channel = 3, output_size=48)

    for checkpoint_file in model_checkpoints:
        checkpoint = torch.load(checkpoint_file)    
        generator.load_state_dict(checkpoint['gen_state_dict'])
        generator.cuda()
        generator.eval()

        fixed_latents = generator.sample_latent(num_images)
        fixed_latents = fixed_latents.cuda()

        num_batches = num_images // batch_size
        fake_images = []
        real_images = []

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            generated_images = generator(fixed_latents[start:end]).cpu().data
            fake_images.append(generated_images.cuda())

        
        for i,data in enumerate(train_loader):
            real_images.append(data[0].cuda())
            
            
        print(len(fake_images), len(real_images))
        fid_value = calculate_fid_given_paths(real_images, fake_images, batch_size, True, dims)
        print('FID: ', fid_value)
        fid_scores.append(fid_value)

    print(len(steps), len(fid_scores))
    plt.plot(steps,fid_scores)
    plt.xlabel('Iterators')
    plt.ylabel('FID')
    plt.savefig('fid_scores.png')



