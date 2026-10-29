import math
import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft
from torchvision.models import vgg19, VGG19_Weights

import utils

# VGGLoss
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()   
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]      

    def forward(self, x, y):
        if x.size(1) == 1 and y.size(1) == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1)
            
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
    
class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 7):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 9):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 12):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# Weighted-rate Loss
class BPPLoss(nn.Module):
    def __init__(self, config):
        super(BPPLoss, self).__init__()
        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B
        self.lambda_schedule = config.lambda_schedule
        self.target_rate = config.target_rate
        self.target_schedule = config.target_schedule

    def forward(self, step_counter, total_nbpp, total_qbpp):
        lambda_A = self.get_scheduled_params(self.lambda_A, self.lambda_schedule, step_counter, ignore_schedule=False)
        lambda_B = self.get_scheduled_params(self.lambda_B, self.lambda_schedule, step_counter, ignore_schedule=False)
        assert lambda_A > lambda_B, f"Expected lambda_A > lambda_B, got (A) {lambda_A} <= (B) {lambda_B}"

        total_qbpp = total_qbpp.item()
        target_bpp = self.get_scheduled_params(self.target_rate, self.target_schedule, step_counter, ignore_schedule=False)
        rate_penalty = lambda_A if total_qbpp > target_bpp else lambda_B
        weighted_rate = rate_penalty * total_nbpp
        return weighted_rate, float(rate_penalty)
    
    def get_scheduled_params(self, param, param_schedule, step_counter, ignore_schedule=False):
        # e.g. schedule = dict(vals=[1., 0.1], steps=[N])
        # reduces param value by a factor of 0.1 after N steps
        if ignore_schedule is False:
            vals, steps = param_schedule['vals'], param_schedule['steps']
            assert(len(vals) == len(steps)+1), f'Mispecified schedule! - {param_schedule}'
            idx = np.where(step_counter < np.array(steps + [step_counter+1]))[0][0]
            param *= vals[idx]
        return param

# R-D Loss
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, quality):
        super().__init__()
        qualities_to_lambda = {
            1: 0.0018,
            2: 0.0035,
            3: 0.0067,
            4: 0.0130,
            5: 0.0250,
            6: 0.0483,
            7: 0.0932,
            8: 0.1800
        }
        self.mse = nn.MSELoss()
        self.lmbda = qualities_to_lambda[quality]

    def forward(self, output, target):
        out = {}
        N, _, H, W = target.size()
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        output_recon = utils.crop_image(output["x_hat"], (880, 1600), stacked_complex=False)
        target_img = utils.crop_image(target, (880, 1600), stacked_complex=False)

        out["mse_loss"] = self.mse(0.95 * output_recon, target_img)
        out["loss"] = (self.lmbda * 255 ** 2 * out["mse_loss"]) + out["bpp_loss"]
        return out

# Distortion Loss
class DistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        out = {}
        output_recon = utils.crop_image(output, (880, 1600), stacked_complex=False)
        target_img = utils.crop_image(target, (880, 1600), stacked_complex=False)

        out["mse_loss"] = self.mse(output_recon, target_img)
        out["loss"] = self.mse(0.95 * output_recon, target_img)
        return out

