# Original code is from https://github.com/HoloCompress/DPRC.githttps://github.com/CompVis/stable-diffusion.git
# Only generation parts are here. compression part is trimmed for simplicity

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.operations import rect_to_polar, polar_to_rect
from .base import BaseNet

# Classes 
##############################################################################
# The Intial Phase Predictor (IP)
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

# The Phase encoder and Phase decoder (Ep and Dp)
class ComPhaseNet(nn.Module):
    def __init__(self, num_in=2, latent_channels=8, num_down=2, num_up=2, 
                 norm=nn.BatchNorm2d, use_dropout=False, activation_last=nn.Hardtanh(-torch.pi, torch.pi)):
        super(ComPhaseNet, self).__init__()

        self.encoder = MultiScale_Encoder(per_layer_out_ch=[32, 64], in_channels=num_in, use_dropout=use_dropout,
                                     out_channels=latent_channels, norm=norm)
        self.decoder = Generator(filters=[64, 32, 16], in_channels=latent_channels, activation='relu', 
                                n_residual_blocks=4)
        self.activation_last = activation_last

    def forward(self, input_x):
        latent_map = self.encode(input_x)
        output = self.decode(latent_map) 
        return latent_map, output
    
    def encode(self, input_x):
        latent_map = self.encoder(input_x)
        return latent_map 

    def decode(self, latent_map):
        z = self.decoder(latent_map)
        poh = self.activation_last(z)
        return poh

# Full Framework
class DPRC(BaseNet):
    """
    Class initialization parameters
    -------------------------------
    distance: propagation distance between the SLM and the target plane 
    wavelength: the wavelength of the laser 
    feature_size: the pixel pitch of the SLM 
    initial_phase:  a module that is used to predict the initial phase at the target plane 
    final_phase_only: the module that is used to encode the complex wavefield at SLM plane into a phase-only hologram
    proptype: chooses the propagation operator. Default ASM.
    linear_conv: if True, pads for linear conv for propagation. Default True
    """
    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        # submodules
        self.initial_phase = InitialPhaseUnet(4, 16, activation_last=self.activation_init)
        self.final_phase_only = ComPhaseNet(num_in=2, activation_last=self.activation_last)
        self.hyper_prior = Hyperprior(
            bottleneck_capacity=8)

        self.devive = device

    def forward(self, target_amp):
        # compute some initial phase, convert to real+imag representation
        init_phase = self.initial_phase(target_amp)
        real, imag = polar_to_rect(target_amp, init_phase)
        target_complex = torch.complex(real, imag)

        # Propagate the wavefield to the SLM plane 
        prop = self.propagator_backward(target_complex)
        slm_field = prop['g_out']

        # Transform it to amplitude-phase form
        amp, ang = rect_to_polar(slm_field.real, slm_field.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)

        # During trainng stage, we didn't conduct entropy coding process
        latent_map = self.final_phase_only.encode(slm_amp_phase)   
        decode_phase = self.final_phase_only.decode(latent_map)

        return {
            'output_poh': decode_phase
        }
      
# networks.py
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, 
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        #norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        norm_kwargs = dict(affine=True)

        if channel_norm is True:
            self.interlayer_norm = ChannelNorm2D_wrap
        else:
            #self.interlayer_norm = instance.InstanceNorm2D_wrap
            self.interlayer_norm = nn.BatchNorm2d

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res) 
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)

class Generator(nn.Module):
    def __init__(self, filters=[960, 480, 240, 120, 60], in_channels=16, activation='relu',
                 n_residual_blocks=8, sample_noise=False, noise_dim=32):

        super(Generator, self).__init__()
        
        kernel_dim = 3
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(affine=True)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = len(filters) - 1

        
        self.interlayer_norm = nn.BatchNorm2d

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)

        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            self.interlayer_norm(in_channels, **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(in_channels, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(in_channels=filters[0], channel_norm=False, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        for i in range(0, len(filters)-1):
            cur_block = nn.Sequential(
                nn.ConvTranspose2d(filters[i], filters[i+1], kernel_dim, **cnn_kwargs),
                self.interlayer_norm(filters[i+1], **norm_kwargs),
                self.activation(),
            )
            setattr(self, 'upconv_block%d'%(i+1), cur_block)


        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 1, kernel_size=(7,7), stride=1),
        )


    def forward(self, x):        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, C, H, W = tuple(head.size())
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head,z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)
        
        x += head
        for k in range(self.n_upsampling_layers):
            cur_layer = getattr(self, 'upconv_block%d'%(k+1))
            x = cur_layer(x)

        out = self.conv_block_out(x)

        return out

class MultiScale_Encoder(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 out_channels=None,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()
        self.per_layer_out_ch = per_layer_out_ch

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.down_conv0 = DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm)
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                cur_layer = [DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm)]
                cur_layer.append(ResidualBlock(in_channels=per_layer_out_ch[i + 1]))
                setattr(self, 'down_conv%d'%(i+1), nn.Sequential(*cur_layer))

            
            self.direct_down1 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=4,
                                  padding=0,
                                  stride=4,
                                  bias=True if norm is None else False))

            self.direct_down2 = nn.Sequential(nn.ReflectionPad2d(0),
                                  nn.Conv2d(in_channels,
                                  per_layer_out_ch[-1],
                                  kernel_size=8,
                                  padding=0,
                                  stride=8,
                                  bias=True if norm is None else False),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))

            smooth_layer = nn.Sequential(
                                Conv2dSame(int(3 * per_layer_out_ch[-1]), per_layer_out_ch[-1], kernel_size=3, bias = None),
                                norm(per_layer_out_ch[-1]),
                                nn.LeakyReLU(0.2))

            setattr(self, 'fusion_layer', smooth_layer)
            out_resLayer = ResidualBlock(in_channels= int(per_layer_out_ch[-1]))
            setattr(self, 'out_res', out_resLayer)

        if  out_channels != None:
            out_layer = Conv2dSame(int(per_layer_out_ch[-1]), out_channels, kernel_size=3, bias = True)
            setattr(self, 'out_layer', out_layer)

    def forward(self, input):
        out_down1 = self.direct_down1(input)
        out_down2 = self.direct_down2(input)
        x = input 
        for k in range(len(self.per_layer_out_ch)):
            cur_layer = getattr(self, 'down_conv%d'%(k))
            x = cur_layer(x)
        cat_features = torch.cat([out_down1, out_down2, x], dim=1)
        out = self.fusion_layer(cat_features)
        out = self.out_res(out)
        out = self.out_layer(out)
        return out

# Hyperprior.py

MIN_SCALE = 0.11
LOG_SCALES_MIN = -3.
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e3
SMALL_HYPERLATENT_FILTERS = 192
LARGE_HYPERLATENT_FILTERS = 64

class CodingModel(nn.Module):
    """
    Probability model for estimation of (cross)-entropies in the context
    of data compression. TODO: Add tensor -> string compression and
    decompression functionality.
    """

    def __init__(self, n_channels, min_likelihood=MIN_LIKELIHOOD, max_likelihood=MAX_LIKELIHOOD):
        super(CodingModel, self).__init__()
        self.n_channels = n_channels
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)

    def _quantize(self, x, mode='noise', means=None):
        """
        mode:       If 'noise', returns continuous relaxation of hard
                    quantization through additive uniform noise channel.
                    Otherwise perform actual quantization (through rounding).
        """

        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise

        elif mode == 'quantize':
            if means is not None:
                x = x - means
                x = torch.floor(x + 0.5)
                x = x + means
            else:
                x = torch.floor(x + 0.5)
        else:
            raise NotImplementedError
        
        return x

    def quantize_latents_st(self, inputs, means=None):
        # Latents rounded instead of additive uniform noise
        # Ignore rounding in backward pass
        values = inputs

        if means is not None:
            values = values - means

        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if means is not None:
            values = values + means

        return values

class Hyperprior(CodingModel):
    def __init__(self, bottleneck_capacity=220, hyperlatent_filters=LARGE_HYPERLATENT_FILTERS, 
        mode='large', scale_lower_bound=MIN_SCALE):

        """
        Introduces probabilistic model over latents of 
        latents.

        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """

        super(Hyperprior, self).__init__(n_channels=bottleneck_capacity)
        
        self.bottleneck_capacity = bottleneck_capacity
        self.scale_lower_bound = scale_lower_bound

        analysis_net = HyperpriorAnalysis
        synthesis_net = HyperpriorSynthesis

        if mode == 'small':
            hyperlatent_filters = SMALL_HYPERLATENT_FILTERS

        self.analysis_net = analysis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        #self.synthesis_mu = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)
        self.synthesis_stats = synthesis_net(C=int(2*bottleneck_capacity), N=hyperlatent_filters)

    def forward(self, latents):

        hyperlatents = self.analysis_net(latents)
        
        # Mismatch b/w continuous and discrete cases?
        # Differential entropy, hyperlatents
        noisy_hyperlatents = self._quantize(hyperlatents, mode='noise')
        hyperlatents_decoded = noisy_hyperlatents

        # Recover latent statistics from compressed hyperlatents
        latent_stats = self.synthesis_stats(hyperlatents_decoded)
        latent_means, _ = torch.split(latent_stats, self.bottleneck_capacity, dim=1)

        latents_decoded = self.quantize_latents_st(latents, latent_means)

        return latents_decoded

class HyperpriorAnalysis(nn.Module):
    """
    Hyperprior 'analysis model' as proposed in [1]. 

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of input channels
    """
    def __init__(self, C=220, N=320, activation='relu'):
        super(HyperpriorAnalysis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.activation = getattr(F, activation)
        self.n_downsampling_layers = 2

        self.conv1 = nn.Conv2d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv2d(N, N, **cnn_kwargs)

    def forward(self, x):
        
        # x = torch.abs(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        return x

class HyperpriorSynthesis(nn.Module):
    """
    Hyperprior 'synthesis model' as proposed in [1]. Outputs 
    distribution parameters of input latents.

    [1] Ballé et. al., "Variational image compression with a scale hyperprior", 
        arXiv:1802.01436 (2018).

    C:  Number of output channels
    """
    def __init__(self, C=220, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose2d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose2d(N, C, kernel_size=3, stride=1, padding=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

# pytorch_prototypes.py

class Identity(nn.Module):
    '''Helper module to allow Downsampling and Upsampling nets to default to identity if they receive an empty list.'''

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input
    
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

# channel.py
def ChannelNorm2D_wrap(input_channels, momentum=0.1, affine=True,
                       track_running_stats=False, **kwargs):
    """
    Wrapper around Channel Norm module
    """
    channel_norm_layer = ChannelNorm2D(input_channels, 
        momentum=momentum, affine=affine,
        track_running_stats=track_running_stats)

    return channel_norm_layer

class ChannelNorm2D(nn.Module):
    """ 
    Similar to default Torch instanceNorm2D but calculates
    moments over channel dimension instead of spatial dims.
    Expects input_dim in format (B,C,H,W)
    """

    def __init__(self, input_channels, momentum=0.1, eps=1e-3,
                 affine=True, **kwargs):
        super(ChannelNorm2D, self).__init__()

        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Image tensor, shape (B,C,H,W)
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta
        return x_normed