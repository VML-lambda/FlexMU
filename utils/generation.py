import torch
import torch.nn as nn
import torch.optim as optim
from utils.operations import crop_image

def generation_sgd(
    init_phase, target_amp, propagator, roi_res,
    num_iters, loss = nn.MSELoss(), lr=8e-3, lr_s=2e-3, s0=1.0):

    device = init_phase.device

    s = torch.tensor(s0, requires_grad=True, device=device)
    slm_phase = init_phase.requires_grad_(True)
 
    optvars = [{'params': slm_phase}]
    if lr_s > 0:
        optvars += [{'params': s, 'lr': lr_s}]
    optimizer = optim.Adam(optvars, lr=lr)

    for k in range(num_iters):
        optimizer.zero_grad()
        slm_field = torch.exp(1j * slm_phase)

        recon_field = propagator(slm_field)['g_out']
        recon_amp = recon_field.abs()
        recon_amp = crop_image(recon_amp, roi_res)
        lossValue = loss(s * recon_amp, target_amp)
        lossValue.backward()
        optimizer.step()

    return slm_phase, recon_amp, lossValue, s