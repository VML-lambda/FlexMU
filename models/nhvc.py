import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import cast

import torchvision
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import get_scale_table
from compressai.ops.parametrizers import NonNegativeParametrizer

from .base import BaseNet

class InitialPhaseUnet(nn.Module):
    def __init__(self, 
                 num_down=4, 
                 num_features_init=16, 
                 max_features=256, 
                 activation_last = nn.Hardtanh(-torch.pi, torch.pi), 
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()
        channel = 1
        net = nn.Sequential(Unet(channel, channel, 
                                 num_features_init, 
                                 max_features,
                                 num_down, 
                                 upsampling_mode='transpose',
                                 use_dropout=False,
                                 norm=norm),
                                 activation_last)
        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase

class NHVC(BaseNet):
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        N = 128
        M = config['model']
        #self.stage = M['stage']
        self.stage = 'only_generation'

        # self.entropy_bottleneck = EntropyBottleneck(N)
        # self.gaussian_conditional = GaussianConditional(None)

        # change forward function by stage
        if self.stage == 'only_generation':
            self.forward = self.generation_forward
        elif self.stage == 'compression':
            self.forward = self.compression_forward

        self.initial_phase_generation = InitialPhaseUnet(
            num_down=4,
            num_features_init=16,
            max_features=256,
            norm=nn.BatchNorm2d,
            activation_last=self.activation_init
        ).to(device)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(2, N, stride=2),
            ResidualBlockDeformable(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlockDeformable(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlockDeformable(N, N),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1)
        ).to(device)

        self.g_s = nn.Sequential(
            ResidualBlockDeformable(N, N),
            ResidualBlockUpsample(N, N, upsample_mode="transposed"),
            ResidualBlockDeformable(N, N),
            ResidualBlockUpsample(N, N, upsample_mode="transposed"),
            ResidualBlockDeformable(N, N),
            ResidualBlockUpsample(N, N, upsample_mode="transposed"),
            ResidualBlockDeformable(N, N),
            upsample_scale_2(N, 1, upsample_mode="transposed"),
            self.activation_last
        ).to(device)

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        ).to(device)

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            upsample_scale_2(N, N, upsample_mode="transposed"),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N*3//2),
            nn.LeakyReLU(inplace=True),
            upsample_scale_2(N*3//2, N*3//2, upsample_mode="transposed"),
            nn.LeakyReLU(inplace=True),
            conv3x3(N*3//2, N*2)
        ).to(device)


    def initial_hologram(self, x):
        init_phase = self.initial_phase_generation(x)
        init_complex = x * torch.exp(1j * init_phase)

        slm = self.propagator_backward(init_complex)['g_out']
        slm_amp_phase = torch.cat((torch.abs(slm), torch.angle(slm)), dim=1)

        return slm_amp_phase

    def generation_forward(self, x):
        slm = self.initial_hologram(x)
        y = self.g_a(slm)
        poh = self.g_s(y)

        return {
            'output_poh' : poh
        }
    
    def compression_forward(self, x):
        slm = self.initial_hologram(x)
        y = self.g_a(slm)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "output_poh": x_hat,
            "likelihoods": {
                "y": y_likelihoods, 
                "z": z_likelihoods
            }
        } 

    def compress(self, x):
        slm = self.initial_hologram(x)
        y = self.g_a(slm)   
        z = self.h_a(y)        
        z_strings = self.entropy_bottleneck.compress(z)                         # Hyperprior "compress"
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])    # Hyperprior decompress
        gaussian_params = self.h_s(z_hat)                   
        scales_hat, means_hat = gaussian_params.chunk(2, 1) 
        indexes = self.gaussian_conditional.build_indexes(scales_hat)               # Set Gaussian Scale
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat) # Latent "compress"
        return {
            "latent_map": y,
            "strings": [y_strings, z_strings], 
            "shape": z.size()[-2:],
        }

    def decompress(self, compressed):
        strings = compressed["strings"]
        shape = compressed["shape"]

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)     # Hyperprior decompress
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)    # Latent feature map decompress

        x_hat = self.g_s(y_hat) # Phase-only Hologram
        return {
            "output_poh": x_hat,
        }

    def update(self, scale_table=None):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table)
        self.entropy_bottleneck.update()
        self.gaussian_conditional.update()

    def aux_loss(self):
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)

class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        super(DownBlock, self).__init__()
        middle_channels = in_channels if middle_channels is None else middle_channels
        bias=True if norm is None else False
        net = list()

        if prep_conv:
            net += [nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
            if norm is not None:
                net += [norm(middle_channels, affine=True)]
            net += [nn.LeakyReLU(0.2, True)]
            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        # Down-sampling
        net += [nn.Conv2d(middle_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=bias)]
        if norm is not None:
            net += [norm(out_channels, affine=True)]
        net += [nn.LeakyReLU(0.2, True)]
        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        # Final Down-block network
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        super(UpBlock, self).__init__()
        bias=True if norm is None else False
        net = list()

        # Up-sampling
        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [nn.Conv2d(in_channels // 4, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
        else:
            raise ValueError("Unknown upsampling mode!")
        if norm is not None:
            net += [norm(out_channels, affine=True)]
        net += [nn.ReLU(True)]
        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        # Convolution Layer
        if post_conv:
            net += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias)]
            if norm is not None:
                net += [norm(out_channels, affine=True)]
            net += [nn.ReLU(True)]
            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        # Final Up-block network
        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        # if skipped is not None:
        #     input = torch.cat([skipped, x], dim=1)
        # else:
        #     input = x
        input = torch.cat([skipped, x], dim=1) if skipped is not None else x
        return self.net(input)
    
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 level: int = None,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super(UnetSkipConnectionBlock, self).__init__()
        self.level = level

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm, upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm, upsampling_mode=upsampling_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        h = self.model(x)
        hs = x
        return torch.cat([hs, h], 1)

class Unet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 max_channels,
                 num_down,   
                 upsampling_mode='transpose',
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d
                 ):
        super(Unet, self).__init__()
        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in-block
        self.in_layer = [nn.Conv2d(in_channels, nf0, kernel_size=3, padding=1, stride=1, bias=True if norm is None else False)]
        self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]
        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center U-net block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels), 
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout, dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)
        for i in list(range(0, num_down - 1))[::-1]:
            level = num_down - 1 - i
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),  
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout, dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      level=level,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out-layer
        self.out_layer = [nn.Conv2d(2 * nf0, out_channels, kernel_size=3, padding=1, stride=1, bias=True)]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer
    

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def upsample_scale_2(in_ch, out_ch, upsample_mode="transposed"):
    if upsample_mode == "transposed":
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, output_padding=0, padding=1)
    
    elif upsample_mode == "pixel_shuffle":
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 2**2, kernel_size=3, padding=1), 
            nn.PixelShuffle(2)
        )
    
    else:   # mode == "nearest", "linear", "bilinear", "bicubic", "trilinear"
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), 
            nn.Upsample(scale_factor=2, mode=upsample_mode),
        )

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_channels, 1*kernel_size*kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator
        )
        return x

class GDN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inverse: bool = False,
                 beta_min: float = 1e-6,
                 gamma_init: float = 0.1):
        super(GDN, self).__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)

        out = x * norm
        return out

class ResidualBlockDeformable(nn.Module):
    """
        Residual deformable block with a stride on the first convolution.

        ResidualBlockDeformable initialization parameters
        -------------------------------
        in_ch (int):    number of input channels
        out_ch (int):   number of output channels
        stride (int):   stride value (default: 2)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super(ResidualBlockDeformable, self).__init__()
        self.conv1 = DeformableConv2d(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(out_ch, out_ch)
        # self.gdn = GDN(out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        # out = self.gdn(out)

        identity = x if self.skip is None else self.skip(x)
        out += identity
        return out

class ResidualBlockWithStride(nn.Module):
    """
        Residual block with a stride on the first convolution.

        ResidualBlockWithStride initialization parameters
        -------------------------------
        in_ch (int):    number of input channels
        out_ch (int):   number of output channels
        stride (int):   stride value (default: 2)
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super(ResidualBlockWithStride, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.skip = conv1x1(in_ch, out_ch, stride) if stride != 1 or in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        identity = x if self.skip is None else self.skip(x)
        out += identity
        return out

class ResidualBlockUpsample(nn.Module):
    """
        Residual block with upsampling on the last convolution.

        ResidualBlockUpsample initialization parameters
        -------------------------------
        in_ch (int):    number of input channels
        out_ch (int):   number of output channels
        upsample (str): upsampling mode (transposed, pixel_shuffle, nearest, bilinear, bicubic)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample_mode: str="transposed"):
        super(ResidualBlockUpsample, self).__init__()
        self.subpel_conv1 = upsample_scale_2(in_ch, out_ch, upsample_mode)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.skip = upsample_scale_2(in_ch, out_ch, upsample_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.subpel_conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.igdn(out)
        identity = self.skip(x)
        out += identity
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
