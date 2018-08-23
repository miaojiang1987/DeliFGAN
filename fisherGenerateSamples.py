from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import models.dcgan as dcgan
import models.mlp as mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--wdecay', type=float, default=1e-6,  help='wdecay value for Phi')
parser.add_argument('--wdecayV', type=float, default=1e-3, help='wdecay value for v')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--hiDiterStart'  , action='store_true', help='do many D iters at start')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--G_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--D_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--rho', type=float, default=1e-6, help='Weight on the penalty term for (sigmas -1)**2')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
images = list()
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw', 'celeba']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)

if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, opt.G_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, opt.G_extra_layers)

if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))

netG.load_state_dict(torch.load("/home/miao/Documents/courses/cv/gan/FisherGAN/samples/netG_epoch_19.pth"))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, opt.D_extra_layers)


if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


mu = np.random.uniform(-1,1,(opt.batchSize,nz,1,1))
mu.fill(0) 
sigma = np.random.uniform(0,0.75,(opt.batchSize,nz,1,1))
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
Lambda = torch.FloatTensor([0]) # lagrange multipliers

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    Lambda = Lambda.cuda()
Lambda = Variable(Lambda, requires_grad=True)

mu = Variable(torch.from_numpy(mu).float(), requires_grad=True) #changed
sigma = Variable(torch.from_numpy(sigma).float(),requires_grad=True) #changed
lambdaG = 1e-5
#mu.data = torch.load('/home/miao/Documents/courses/cv/gan/FisherGAN/samples/mu_epoch_19.pth')
#sigma.data = torch.load('/home/miao/Documents/courses/cv/gan/FisherGAN/samples/sigma_epoch_19.pth')
print("SIGMA MEAN : " +str(sigma.data.mean()))
print("SIGMA VARIANCE : " +str(sigma.data.var()))
# setup optimizer
paramsD = [{'params': netD.phi.parameters(),'weight_decay': opt.wdecay }, 
           {'params': netD.v.parameters(),  'weight_decay': opt.wdecayV}]
if opt.adam:
    optimizerD = optim.Adam(paramsD,           lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(paramsD,           lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0
for epoch in range(3000):
	print("EPOCH :"+str(epoch))
	epsilon = np.random.normal(0,1,(opt.batchSize,nz,1,1))
	epsilon = torch.from_numpy(epsilon).float()
	epsilon = Variable(epsilon,requires_grad=False)
	epsilon.data.normal_(0,1)

#	noise = mu + sigma*epsilon
	noise.resize_(opt.batchSize, nz, 1, 1).normal_(0,1)
	noisev = Variable(noise)
	gen_iterations += 1

#	print('MU : '+ (str)(mu.data.mean()) +' ' +(str)(mu.data.std()))
#	print('SIGMA : '+ (str)(sigma.data.mean()) +' ' +(str)(sigma.data.std()))

	fake = netG(noisev)
#	fake = netG(Variable(noise.data.cpu, volatile=True))
	fake.data = fake.data.mul(0.5).add(0.5)
#	images.append(fake.data)	
	vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    
