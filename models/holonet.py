# Original Code is from https://github.com/computational-imaging/neural-holography
# This Code has some refactoring for our convinience

"""
This is the script that is used for the implementation of HoloNet. The class HoloNet(nn.module) is 
described in the following.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import math
import torch
import torch.nn as nn

from utils.operations import polar_to_rect, rect_to_polar

from .base import BaseNet

class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, 
                 num_down=8, 
                 num_features_init=32, 
                 max_features=256,
                 activation_last = nn.Hardtanh(-torch.pi, torch.pi), 
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
                    activation_last]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase

class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=4, activation_last=nn.Hardtanh(-math.pi, math.pi)):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
                    activation_last]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase

class PhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a target amplitude"""
    def __init__(self, num_down=10, num_features_init=16, norm=nn.BatchNorm2d):
        super(PhaseOnlyUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, 1024,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, target_amp):
        out_phase = self.net(target_amp)
        return (torch.ones(1), out_phase)

class HoloNet(BaseNet):
    """Generates phase for the final non-iterative model

    Class initialization parameters
    -------------------------------
    distance: propagation dist between SLM and target, in meters, default 0.1.
        Note: distance is negated internally, so the PhaseGenerator and
        ProcessAndPropagate get the same input
    wavelength: the wavelength of interest, in meters, default 520e-9
    feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    initial_phase: a module that returns an initial phase given the target amp.
        Default None, which assumes all zeros initial phase
    final_phase_only: a module that processes the post-propagation amp+phase to
        a phase-only output that works as well as iterative results. Default
        None, which switches to double phase coding
    linear_conv: if True, pads for linear conv for propagation. Default True

    Usage
    -----
    Functions as a pytorch module:

    >>> phase_generator = HoloNet(...)
    >>> slm_amp, slm_phase = phase_generator(target_amp)

    target_amp: amplitude at the target plane, with dimensions [batch, 1,
        height, width]
    slm_amp: amplitude to be encoded in the phase pattern at the SLM plane. Used
        to enforce uniformity, if desired. Same as target dimensions
    slm_phase: encoded phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        # submodules
        self.initial_phase = InitialPhaseUnet(4, 16, activation_last=self.activation_init)
        self.final_phase_only = FinalPhaseOnlyUnet(4, 16, num_in=2, activation_last=self.activation_last)


    def forward(self, target_amp):
        # compute some initial phase, convert to real+imag representation
        init_phase = self.initial_phase(target_amp)
        real, imag = polar_to_rect(target_amp, init_phase)
        target_complex = torch.complex(real, imag)

        # implement the basic propagation to the SLM plane
        prop = self.propagator_backward(target_complex)
        slm_naive = prop['g_out']
        
        # switch to amplitude+phase and apply source amplitude adjustment
        amp, ang = rect_to_polar(slm_naive.real, slm_naive.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)
        output_poh = self.final_phase_only(slm_amp_phase)

        return {
            'output_poh': output_poh
        }

# Modules
class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=True if norm is None else False)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [
                Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
                           bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.ReLU(True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        if post_conv:
            net += [Conv2dSame(out_channels,
                               out_channels,
                               kernel_size=3,
                               bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels, affine=True)]

            net += [nn.ReLU(True)]

            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)

class DownBlock(nn.Module):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        if middle_channels is None:
            middle_channels = in_channels

        net = list()

        if prep_conv:
            net += [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels,
                              middle_channels,
                              kernel_size=3,
                              padding=0,
                              stride=1,
                              bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(middle_channels, affine=True)]

            net += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        net += [nn.ReflectionPad2d(1),
                nn.Conv2d(middle_channels,
                          out_channels,
                          kernel_size=4,
                          padding=0,
                          stride=2,
                          bias=True if norm is None else False)]

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class UnetSkipConnectionBlock(nn.Module):
    '''Helper class for building a 2D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)

class Unet(nn.Module):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 use_dropout,
                 upsampling_mode='transpose',
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        self.in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout,
                                                  dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout,
                                                      dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv2dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear or (norm is None))]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]

            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


