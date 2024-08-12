# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
import kornia.filters as KF
# from loss_ssim import ssim
shape = (256, 256)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp






class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def forward(self, I, J, win=[15]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims
        sum_filt = torch.ones([1, I.shape[1], *win]).cuda()/I.shape[1]
        pad_no = math.floor(win[0] / 2)
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)
        cc = cross * cross / ((I_var * J_var).clamp(min=1e-3) + 1e-3)
        return -1 * torch.mean(cc)

class mi_loss(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(mi_loss, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))

        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)

def l1loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,(3,3),(1,1))*mask_
    img2 = KF.gaussian_blur2d(img2,(3,3),(1,1))*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).mean()


def l2loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,(3,3),(1,1))*mask_
    img2 = KF.gaussian_blur2d(img2,(3,3),(1,1))*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss,self).__init__()
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()

    def forward(self,img1,img2,mask=1,eps=1e-2):
        #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        #l = l[...,5:-5,10:-10].mean()
        return l

def smoothloss(disp,img=None):
    smooth_d=[3*3,7*3,15*3]
    b,c,h,w = disp.shape
    grad = KF.spatial_gradient(disp,order=2).abs().sum(dim=2)[:,:,5:-5,5:-5].clamp(min=1e-9).mean()
    local_smooth_re = 0
    for d in smooth_d:
        local_mean = KF.gaussian_blur2d(disp,(d,d),(d//6,d//6),border_type='replicate')
        #local_mean_pow2 = F.avg_pool2d(disp.pow(2),kernel_size=d,stride=1,padding=d//2)
        local_smooth_re += 1/(d*1.0+1)*(disp-local_mean)[:,:,d//2:-d//2,d//2:-d//2].pow(2).mean()
        #local_smooth_re += 1/(d*1.0+1)*(disp.pow(2)-local_mean_pow2)[:,:,5:-5,5:-5].pow(2).mean()
    #global_var = disp[...,2:-2,2:-2].var(dim=[-1,-2]).clamp(1e-5).mean()
    #std = img.std(dim=[-1,-2]).mean().clamp(min=0.003)
    #grad = grad[...,10:-10,10:-10]
    return 5000*local_smooth_re + 500*grad

def l2regularization(img):
    return img.pow(2).mean()



