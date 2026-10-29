from typing import Union
import torch.fft as fft
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.operations import (pad_image, crop_image)

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# COORDINATE
# Every coordinate in this code follows this x, y rule
# O is left-upper origin
# O → y 
# ↓
# x

def kernel_factory(prop_config: dict, device: torch.device):
    P = prop_config
    isQE         = P['isQE']
    wavelength   = P['wavelength'] * nm
    feature_size = (P['feature_size'][0] * um, 
                    P['feature_size'][1] * um)
    prop_dist    = P['prop_dist'] * cm
    field_res    = P['field_res']

    kernel_base = asm_kernel(field_res ,feature_size, wavelength, isQE, prop_dist, device)
    mask = torch.ones_like(kernel_base, dtype=torch.float)

    if P.get('isBandLimit', False):
        mask = mask * bl_mask(field_res ,feature_size, wavelength, isQE, prop_dist).to(device)

    if P.get('isAperture', False):
        A = P['aperture_size']
        mask = mask * ellipse_binary_mask(A['resolution'] ,A['center'], A['radius']).to(device)

    if P.get('isDCMasking', False):
        D = P['dc_size']
        mask = mask * (1 - ellipse_binary_mask(D['resolution'] ,D['center'], D['radius']).to(device))

    if P.get('isMaskSmoothing', False):
        mask = gaussian_smoothing(mask)

    return kernel_base * mask

def asm_kernel(field_resolution, feature_size, wavelength, isQE, z, device):
    conv_size = [i * 2 for i in field_resolution] if isQE else field_resolution
    num_x, num_y = conv_size[-2], conv_size[-1]
    dx, dy = feature_size
    du, dv = 1 / (float(num_x) * dx), 1 / (float(num_y) * dy)

    fx = torch.linspace(-1 / (2 * dx) + du / 4.0, 1 / (2 * dx) - du / 4.0, num_x)
    fy = torch.linspace(-1 / (2 * dy) + dv / 4.0, 1 / (2 * dy) - dv / 4.0, num_y)
    fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

    kernel = torch.exp(1j * 2 * torch.pi * z * torch.sqrt(1 / wavelength**2 - (fxx**2 + fyy**2)))
    kernel = torch.reshape(kernel, (1, 1, *kernel.size()))
    kernel = kernel.to(device)

    return kernel

def bl_mask(field_resolution, feature_size, wavelength, isQE, z):
    conv_size = [i * 2 for i in field_resolution] if isQE else field_resolution
    num_x, num_y = conv_size[-2], conv_size[-1]
    dx, dy = feature_size
    du, dv = 1 / (float(num_x) * dx), 1 / (float(num_y) * dy)

    fx = torch.linspace(-1 / (2 * dx) + du / 4.0, 1 / (2 * dx) - du / 4.0, num_x)
    fy = torch.linspace(-1 / (2 * dy) + dv / 4.0, 1 / (2 * dy) - dv / 4.0, num_y)
    fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')

    # band-limited ASM - Matsushima et al. (2009)
    blx = 1 / math.sqrt((2 * z * du)**2 + 1) / wavelength
    bly = 1 / math.sqrt((2 * z * dv)**2 + 1) / wavelength
    bl_mask = (torch.abs(fxx) < blx) & (torch.abs(fyy) < bly)

    return bl_mask

def ellipse_binary_mask(field_resolution: Union[int, int], center: Union[int, int], radius: Union[int, int]):
    height, width = field_resolution
    center_x, center_y = center
    radius_x, radius_y = radius
    x_grid, y_grid  = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    mask = torch.zeros(field_resolution)
    ellipse_mask = (((x_grid - center_x) ** 2) / radius_x ** 2 + 
                    ((y_grid - center_y) ** 2) / radius_y ** 2) <= 1

    mask[ellipse_mask] = 1
    return mask

def shifted_asm_kernel(field_resolution, feature_size, wavelength, z, shift, device):

    conv_size = [i * 2 for i in field_resolution]
    num_y, num_x = conv_size[-2], conv_size[-1]
    dy, dx = feature_size
    dv, du = 1 / (float(num_y) * dy), 1 / (float(num_x) * dx)

    fy = torch.linspace(-1 / (2 * dy) + dv / 4.0, 1 /
                        (2 * dy) - dv / 4.0, num_y)
    fx = torch.linspace(-1 / (2 * dx) + du / 4.0, 1 /
                        (2 * dx) - du / 4.0, num_x)
    fyy, fxx = torch.meshgrid(fy, fx, indexing='ij')

    kernel = torch.exp(1j * 2 * torch.pi * z * torch.sqrt(1 / wavelength**2 - (fyy**2 + fxx**2))) * torch.exp(1j*2*torch.pi*(shift[0]*fyy+shift[1]*fxx))

    kernel = torch.reshape(kernel, (1, 1, *kernel.size()))
    kernel = kernel.to(device)

    return kernel

def shifted_bl_asm_kernel(field_resolution, feature_size, wavelength, z, shift, device):

    conv_size = [i * 2 for i in field_resolution]
    num_y, num_x = conv_size[-2], conv_size[-1]
    dy, dx = feature_size
    dv, du = 1 / (float(num_y) * dy), 1 / (float(num_x) * dx)

    fy = torch.linspace(-1 / (2 * dy) + dv / 4.0, 1 /
                        (2 * dy) - dv / 4.0, num_y)
    fx = torch.linspace(-1 / (2 * dx) + du / 4.0, 1 /
                        (2 * dx) - du / 4.0, num_x)
    fyy, fxx = torch.meshgrid(fy, fx, indexing='ij')

    kernel = torch.exp(1j * 2 * torch.pi * z * torch.sqrt(1 /
                       wavelength**2 - (fyy**2 + fxx**2))) * torch.exp(1j*2*torch.pi*(shift[0]*fyy+shift[1]*fxx))

    kernel = torch.reshape(kernel, (1, 1, *kernel.size()))

    w_y = 1 / (2 * du) 
    bly_minus = 1 / math.sqrt((z / (shift[0] - w_y))**2 + 1) / wavelength
    bly_plus = 1 / math.sqrt((z / (shift[0] + w_y))**2 + 1) / wavelength
    blx = 1 / math.sqrt((2 * z * du)**2 + 1) / wavelength

    bl_filter = (-bly_minus < fyy) & (fyy < bly_plus) & (torch.abs(fxx) < blx)
    kernel = (kernel * bl_filter)

    kernel = kernel.to(device)

    return kernel

def gaussian_kernel(kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d /= gauss_1d.sum()  

    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    return gauss_2d

def gaussian_smoothing(input_tensor, kernel_size=5, sigma=1.0):
    device = input_tensor.device
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).to(device)
    smoothed = F.conv2d(input_tensor, kernel, padding=kernel_size // 2)
    return smoothed

class PropagatorBase(nn.Module):
    def __init__(self, H, isQE = True):
        super(PropagatorBase, self).__init__()
        self.H = H
        self.isQE = isQE
        self.input_res = [None, None]

    def apply_quadruple_extension(self,x):
        self.input_res = x.size()[-2:]
        conv_size = [i * 2 for i in self.input_res]
        return pad_image(x, conv_size, padval=0)

    def revert_quadruple_extension(self,x):
        return crop_image(x, self.input_res)

    def fourier_transform(self,x):
        return fft.fftshift(fft.fftn(fft.fftshift(x), dim=(-2, -1), norm='ortho'))
    
    def inverse_fourier_transform(self,x):
        return fft.fftshift(fft.ifftn(fft.fftshift(x), dim=(-2, -1), norm='ortho'))

    def forward(self, g_in):
        g_in = self.apply_quadruple_extension(g_in) if self.isQE else g_in
        G_in = self.fourier_transform(g_in)

        assert G_in.shape[-2:] == self.H.shape[-2:], f"G_in and H size doesn't match with {G_in.shape[-2:]}, {self.H.shape[-2:]}"
        G_out = G_in * self.H
        
        g_out = self.inverse_fourier_transform(G_out)
        g_out = self.revert_quadruple_extension(g_out) if self.isQE else g_out

        return {
            'g_out': g_out,
            'G_in' : G_in,
            'G_out': G_out
        }

class PropagatorROI(PropagatorBase):
    def __init__(self, H, roi):
        """ roi : (y_start, x_start, height, width) """
        super(PropagatorROI, self).__init__(H)
        self.input_res = [None, None]
        self.roi = roi 

    def forward(self, g_in):
        y_s, x_s, h, w = self.roi

        g_out = super(PropagatorROI, self).forward(g_in)
        g_out = g_out[:, :, y_s:y_s+h, x_s:x_s+w]
        
        return g_out

class PropagatorShifted(PropagatorBase):
    def __init__(self, H, shift):
        super(PropagatorShifted, self).__init__(H)
        self.input_res = [None, None]
        self.shift = shift

    def fourier_transform(self, x):
        y_base = math.floor(x.size(-2) / 2.0)
        x_base = math.floor(x.size(-1) / 2.0)
        shift_amount = (y_base + self.shift[0], x_base + self.shift[1])

        G = fft.fftn(fft.fftshift(x.type(torch.float32)), dim=(-2, -1), norm='ortho')
        G = torch.roll(G, shifts=shift_amount, dims=(-2,-1))
        return G
    
    def inverse_fourier_transform(self, x):
        y_base = math.floor(x.size(-2) / 2.0)
        x_base = math.floor(x.size(-1) / 2.0)
        shift_amount = (-(y_base + self.shift[0]), -(x_base + self.shift[1]))

        G = torch.roll(x, shifts=shift_amount, dims=(-2,-1))
        G = fft.fftshift(fft.ifftn(G, dim=(-2, -1), norm='ortho'))
        return G
    
class PropagatorFourier(nn.Module):
    def __init__(self):
        super(PropagatorFourier, self).__init__()
        self.input_res = [None, None]

    def apply_quadruple_extension(self,x):
        self.input_res = x.size()[-2:]
        conv_size = [i * 2 for i in self.input_res]
        return pad_image(x, conv_size, padval=0)
    
    def revert_quadruple_extension(self,x):
        return crop_image(x, self.input_res)

    def fourier_transform(self,x):
        return fft.fftshift(fft.fftn(fft.fftshift(x.type(torch.float32)), dim=(-2, -1), norm='ortho'))
    
    def forward(self, x):
        x = self.apply_quadruple_extension(x)
        x = self.fourier_transform(x)
        x = self.revert_quadruple_extension(x)
        return x
    
class PropagatorHDO(PropagatorBase):
    def __init__(self, H, feature_size):
        super(PropagatorHDO, self).__init__(H)
        self.input_res = H.shape[-2:]

        num_y, num_x = self.input_res[-2]//2, self.input_res[-1]//2
        dy, dx = (feature_size, feature_size)

        vy = torch.linspace(-(num_y//2)*dy , ((num_y//2) + (num_y%2))*dy, num_y)
        vx = torch.linspace(-(num_x//2)*dx , ((num_x//2) + (num_x%2))*dx, num_x)

        vyy, vxx = torch.meshgrid(vy, vx, indexing='ij')
        self.shift_mask  = torch.exp(1j*torch.pi*vyy/dy).to(H.device)

    def forward(self, g_in):
        g_in = g_in*self.shift_mask
        g_in = self.apply_quadruple_extension(g_in) if self.isQE else g_in
        G_in = self.fourier_transform(g_in)

        assert G_in.shape[-2:] == self.H.shape[-2:], "G_in and H size doesn't match"
        G_out = G_in * self.H
        
        g_out = self.inverse_fourier_transform(G_out)
        g_out = self.revert_quadruple_extension(g_out) if self.isQE else g_out

        return {
            'g_out': g_out,
            'G_in' : G_in,
            'G_out': G_out
        }
   
