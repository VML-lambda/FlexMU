"""
This file provides uitlity funtions and is borrowed from the implementation of the following paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def pad_image(field, target_shape, padval=0, mode='constant'):
    size_diff = np.array(target_shape[-2:]) - np.array(field.shape[-2:])
    odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        pad_left = int(pad_front[1])
        pad_right = int(pad_end[1])
        pad_top = int(pad_front[0])
        pad_bottom = int(pad_end[0])
        
        if isinstance(field, np.ndarray):
            pad_axes = [(0, 0) for _ in range(field.ndim - 2)] # no padding for non-spatial dimensions
            pad_axes.extend([(pad_top, pad_bottom), (pad_left, pad_right)])
            return np.pad(field, pad_axes, mode=mode, constant_values=padval)
        
        elif isinstance(field, torch.Tensor):
            pad_axes = [pad_left, pad_right, pad_top, pad_bottom]
            return F.pad(field, pad_axes, mode=mode, value=padval)
        
        else:
            raise TypeError("Unsupported type for field. Expected np.ndarray or torch.Tensor.")

    else:
        return field

def crop_image(field, target_shape):
    size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
    odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + odd_dim) // 2
        crop_end = (crop_total + 1 - odd_dim) // 2

        crop_left = int(crop_front[1])
        crop_right = int(crop_end[1])
        crop_top = int(crop_front[0])
        crop_bottom = int(crop_end[0])

        if isinstance(field, np.ndarray):
            return field[..., crop_top:field.shape[-2] - crop_bottom, crop_left:field.shape[-1] - crop_right]

        elif isinstance(field, torch.Tensor):
            return field[..., crop_top:field.shape[-2] - crop_bottom, crop_left:field.shape[-1] - crop_right]

        else:
            raise TypeError("Unsupported type for field. Expected np.ndarray or torch.Tensor.")

    else:
        return field

def pad_crop_to_res(image, target_res):
    padded_img = pad_image(image, target_res)
    cropped_img = crop_image(padded_img, target_res)
    return cropped_img

def ellipse_binary_mask(height, width, radius):
    radius_y, radius_x = radius[-2], radius[-1]
    center_x, center_y = width // 2, height // 2
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    distance = (x_grid ** 2 + y_grid ** 2) ** 0.5
    mask = torch.zeros_like(distance)
    ellipse_mask = (((x_grid - center_x) ** 2) / radius_x ** 2 + 
                    ((y_grid - center_y) ** 2) / radius_y ** 2) <= 1

    mask[ellipse_mask] = 1
    return mask

def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out

def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out
 
def phasemap_8bit(phasemap: torch.Tensor, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed
    """
    out_phase = phasemap.cpu().detach().squeeze().numpy()
    out_phase = ((out_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)

    if inverted:
        phase_out_8bit = ((1 - out_phase) * 255).round().astype(np.uint8) 
    else:
        phase_out_8bit = ((out_phase) * 255).round().astype(np.uint8)  

    return phase_out_8bit

def magnitude_spectrum(field: torch.Tensor):
    input_res = field.size()[-2:]
    conv_size = [i * 2 for i in input_res]

    field_2x = pad_image(field, conv_size, padval=0)
    freq = torch.fft.fft2(field_2x)
    freq = torch.fft.fftshift(freq, dim=(-2, -1))

    magnitude = torch.abs(freq)
    log_magnitude = torch.log1p(magnitude)

    log_magnitude_np = log_magnitude.squeeze().cpu().numpy()
    log_magnitude_np -= log_magnitude_np.min()  
    log_magnitude_np /= log_magnitude_np.max()  
    log_magnitude_np *= 255                    
    log_magnitude_np = log_magnitude_np.astype(np.uint8)

    return log_magnitude_np


def SCD(A: torch.Tensor, B: torch.Tensor, cycle_size=1.0):
    """
        shortest circular distance
        default cycle_size is 1.0 which is default cycle size in tensor holography

        returns A - B in phase space.
        it means that 'B + result = A'
    """

    A_scaled = (A / cycle_size) * 2.0 * torch.pi
    B_scaled = (B / cycle_size) * 2.0 * torch.pi

    phase_diff = torch.atan2(torch.sin(A_scaled - B_scaled),
                             torch.cos(A_scaled - B_scaled))

    unscaled = phase_diff * cycle_size / (2.0 * torch.pi) 

    return unscaled