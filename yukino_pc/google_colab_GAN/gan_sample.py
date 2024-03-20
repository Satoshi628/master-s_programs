import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
from IPython.display import HTML



#生成器でtanhを使っているため入力の範囲を[-1, 1]にする。
class Mydataset(Dataset):
    def __init__(self, mnist_data):
        self.mnist = mnist_data
    def __len__(self):
        return len(self.mnist)
    def __getitem__(self, idx):
        X = self.mnist[idx][0]
        X = (X * 2) - 1
        y = self.mnist[idx][1]
        
        return X, y

#####input#####
b_size = 128
ngpu = 1
nz = 100
lr = 0.0002
num_epochs = 500
###############

#MNISTのデータロード
#mnist_data = datasets.MNIST('../data/MNIST',  transform = transforms.Compose([ transforms.ToTensor()]), download=True)
#torchvision.datasets.FashionMNIST(root, train=True, transform=None, download=False)
fashion_mnist_data = datasets.FashionMNIST('../data/FashionMNIST',  transform = transforms.Compose([ transforms.ToTensor()]), download=True)

mnist_data_norm = Mydataset(fashion_mnist_data)
dataloader = torch.utils.data.DataLoader(mnist_data_norm, batch_size=b_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
#dataloader = torch.utils.data.DataLoader(fashion_mnist_data, batch_size=b_size, shuffle=True, num_workers=2, pin_memory=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#generatorの定義
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (64*8) x 3 x 3
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (64*4) x 6 x 6
            nn.ConvTranspose2d( 128, 128, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (64*2) x 10 x 10
            nn.ConvTranspose2d( 128, 64, 2, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 16 x 16
            nn.ConvTranspose2d( 64, 1, 2, 2, 2, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)




#discriminatorの定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, 64, 2, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 16 x 16
            nn.Conv2d(64, 128, 2, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 10 x 10
            nn.Conv2d(128, 128, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 6 x 6
            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 3 x 3
            nn.Conv2d(128, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



netG = Generator().cuda()
netG.apply(weights_init)

netD = Discriminator().cuda()
netD.apply(weights_init)

#損失関数の定義
criterion = nn.BCELoss()

#可視化用ノイズの定義
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

#教師データのラベル
real_label = 1
fake_label = 0

#更新手法
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


for inputs, _ in dataloader:
    true_label = inputs.new_ones(inputs.shape[0]).cuda(device, non_blocking=True)
    fake_label = inputs.new_zeros(inputs.shape[0]).cuda(device, non_blocking=True)

# トレーニングループ

# 進捗を記録するためのリスト
best_errG = 9999.
img_list = []
G_losses = []
D_losses = []
iters = 0
netG.train()
netD.train()
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, true_label)
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, fake_label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        
        # 勾配を0に
        optimizerD.zero_grad()
        errD = errD_real + errD_fake
        errD.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, true_label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        if errG < best_errG:
            best_errG = errG
            PATH = "modelG.pth"
            torch.save(netG.state_dict(), PATH)
            
        iters += 1



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()