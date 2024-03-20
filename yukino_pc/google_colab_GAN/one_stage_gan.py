import os
import time
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
from torchvision import datasets, transforms
from torchvision.utils import make_grid


logger = None
cfg = None
num_iter_global = 1

class GradientHook(object):
    def __init__(self, module, is_negate=True):
        self.module = module
        self.is_negate = is_negate
        self.handle = None

    def set_negate(self, is_negate):
        self.is_negate = is_negate

    def negate_grads_func(self, module, grad_input, grad_output):
        # if is_negate is false, not negate grads
        if not self.is_negate:
            return

        if isinstance(grad_input, tuple):
            result = []
            for item in grad_input:
                if isinstance(item, torch.Tensor):
                    result.append(torch.neg(item))
                else:
                    result.append(item)
            return tuple(result)
        if isinstance(grad_input, torch.Tensor):
            return torch.neg(grad_input)

    def set_negate_grads_hook(self):
        self.handle = self.module. \
            register_backward_hook(self.negate_grads_func)

    def __del__(self):
        if self.handle:
            self.handle.remove()

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DCGenerator(nn.Module):
    def __init__(self, zdim=100, num_channel=1, d=128):
        super(DCGenerator, self).__init__()
        self.zdim = zdim
        self.deconv1 = nn.ConvTranspose2d(zdim, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, num_channel, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x


class DCDiscriminator(nn.Module):
    def __init__(self, num_channel=1, d=128):
        super(DCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.squeeze(self.conv5(x))

        return x

def train_epoch_adv(generator, discriminator, train_loader, optimizers, cfg, hooks=None):
    generator.cuda().train()
    discriminator.cuda().train()

    optimizer_g, optimizer_d = optimizers


    global num_iter_global
    loss_adv_value = 0.0

    for i, (x, _) in enumerate(train_loader):
        batch_size = x.size(0)
        x = x.cuda()

        y_real = torch.ones(batch_size).cuda()
        y_fake = torch.zeros(batch_size).cuda()

        z = torch.randn((batch_size, 100, 1, 1)).cuda()
        fake = generator(z)

        pred_real = discriminator(x).flatten()
        pred_fake = discriminator(fake).flatten()

        loss_real = F.binary_cross_entropy_with_logits(pred_real, y_real)
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

        loss_adv = loss_real + loss_fake

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        loss_adv.backward()

        loss_val = loss_adv.item()
        loss_real_val = loss_real.item()
        loss_fake_val = loss_fake.item()

        optimizer_g.step()
        optimizer_d.step()

        loss_adv_value += loss_adv.item() * batch_size

        num_iter_global += 1

    loss_adv_value /= len(train_loader.dataset)

    return loss_adv_value, None


def train():
    transform = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_loader = DataLoader(
        datasets.MNIST("/mnt/kamiya/code/google_colab_GAN/MNIST", train=True, transform=transform),
        128, shuffle=True, num_workers=2)

    # ---------------- Network ----------------
    generator = DCGenerator()
    discriminator = DCDiscriminator()

    # ---------------- Set Hook Func ----------------
    hook_grad = GradientHook(generator)
    hook_grad.set_negate_grads_hook()

    train_epoch = train_epoch_adv
    hooks_bn = None

    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    # ---------------- Optimizer ----------------
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0000135, betas=(0.5, 0.999))

    # ---------------- Training ----------------
    z_fix = torch.randn((100, 100, 1, 1)).cuda()

    for epoch in range(1, 100 + 1):
        loss_adv, loss_bn = train_epoch(generator, discriminator, train_loader,
                            (optimizer_g, optimizer_d), cfg, (hook_grad, hooks_bn))
        
        print(loss_adv, loss_bn)
        #end one epoch
        with torch.no_grad():
            fake = generator(z_fix).detach().cpu()
        img = make_grid(fake, padding=2, normalize=True).numpy().transpose(1, 2, 0)
        img = (img + 1.0) / 2 * 255
        cv2.imwrite(f"result/{epoch:04}.png", img)



if __name__ == '__main__':
    train()