"""Diffusion U-Net"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lpips

from model.degradation_encoder import DEG_VEC_DIM


# 1. Timestep Embedding

class TimestepEmbedding(nn.Module):
  '''
  Encodes scalar timestep t into a vector the U-Net can use.
  Uses sinusoidal embedding

  Input:
    t(B,) integer timestep per sample in batch
  Output:
    (B, embed_dim)
  '''
  def __init__(self, embed_dim):
    super().__init__()
    self.embed_dim = embed_dim

    # Small MLP to project sinusoidal embedding to final dim
    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, embed_dim * 4),
        nn.SiLU(),
        nn.Linear(embed_dim * 4, embed_dim),
    )

  def forward(self, t):
    # Sinusiudal embedding - same idea as positional encoding in transformers
    half = self.embed_dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:,None].float() * freqs[None] #(B, half)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (B, embed_dim)

    return self.mlp(emb)


# 2. ResBlock - Basic U-Net building block

class ResBlock(nn.Module):
  '''
  Residual block with timestep conditioning
  Timestep embedding is added after first conv to modulate features.

  Input:
    x:      (B, in_channels, H, W)
    t_emb:  (B, embed_dim)
  Output:
    (B, out_channels, H, W)
  '''
  def __init__(self, in_channels, out_channels, embed_dim):
    super().__init__()

    # Calulate valid num_groups
    def get_groups(channels):
      for g in [8, 4, 2, 1]:
        if channels % g == 0:
          return g

    self.norm1 = nn.GroupNorm(get_groups(in_channels), in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    #projects timestep embedding to out_channels so it can be added to features
    self.t_proj = nn.Linear(embed_dim,  out_channels)

    self.norm2 = nn.GroupNorm(get_groups(out_channels), out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    self.act = nn.SiLU()

    # skip connection - if channels differ, use 1x1 conv to match
    if in_channels != out_channels:
      self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.skip = nn.Identity()

  def forward(self, x, t_emb):
    h= self.conv1(self.act(self.norm1(x)))

    # add timestep info, reshape t_emb to (B, C, 1, 1) for broadcasting
    h = h + self.t_proj(self.act(t_emb))[:, :,  None, None]

    h = self.conv2(self.act(self.norm2(h)))

    return h + self.skip(x) # residual connection


# 3. Diffusion U-Net

class DiffusionUNet(nn.Module):
  '''
  U-Net that predicts noise for the diffusion reverse prcess.
  Conditioned on timestep and multi-sclae context from your pipeline

  Input:
    x_t:      (B, 3, H, W)       noisy image at timestep t
    t:        (B,)               timestep
    context:  dict of tensors    multi-scale context from gating module
      scale_4:  (B, 192, H/4, W/4)
      sclae_8:  (B, 384, H/8, W/8)
      scale_16: (B, 768, H/16, W/16)
  Output:
    nopise: (B, 3, H, W)          predicted noise
  '''
  def __init__(self, embed_dim=256):
    super().__init__()

    self.t_emb = TimestepEmbedding(embed_dim)
    # Projects the 5-layer classifier vector into the timestep embedding space.
    # Added to t_emb so every ResBlock receives degradation-aware conditioning
    # without any per-class branches (AirNet/MCDRNet style).
    self.deg_proj = nn.Linear(DEG_VEC_DIM, embed_dim)

    # Encoder (down sampling)
    # Level 1: Full resolution (H, W)
    # input channels = 3 (noisy image) + 3 (hazy key frame anchor)
    self.enc1 = ResBlock(6, 64, embed_dim)

    # Level 2: H/2, W/2
    # Input channels = 64  + 192 (context scale_4 downsampled)
    self.down1 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
    self.enc2 = ResBlock(64 + 192, 128, embed_dim)

    # Level 3: H/4, W/4
    # Input channels = 128 + 192 (Context scale_8 downsampled)
    self.down2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)
    self.enc3 = ResBlock(128 + 384, 256, embed_dim)

    # Level 4: H/8, W/8
    # Input channels = 256 + 768 (Context scale_16 downsampled)
    self.down3 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
    self.enc4 = ResBlock(256 + 768, 512, embed_dim)

    # Bottle Neck
    self.down4 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
    self.bottleneck = ResBlock(512, 512, embed_dim)

    # Decoder (Upsampling)
    self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
    self.dec4 = ResBlock(512 + 512, 256, embed_dim) # 512 skip + 512 up

    self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
    self.dec3 = ResBlock(256 + 256, 128, embed_dim)

    self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
    self.dec2 = ResBlock(128 + 128, 64, embed_dim)

    self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
    self.dec1 = ResBlock(64 + 64, 64, embed_dim)

    # Output head
    self.out_norm = nn.GroupNorm(8, 64)
    self.out_conv = nn.Conv2d(64, 3, kernel_size=1) # predict 3 channel noise

  def forward(self, x_t, t, context, key_frame, deg_vec=None):
    t_emb = self.t_emb(t)  # (B, embed_dim)

    # Add 5-layer degradation vector to the timestep embedding so every
    # ResBlock is degradation-aware (AirNet/MCDRNet style).
    if deg_vec is not None:
        t_emb = t_emb + self.deg_proj(deg_vec)

    x = torch.cat([x_t, key_frame], dim=1)  # (B, 6, H, W)
    H, W = x.shape[2], x.shape[3]

    # Resize multi-scale context to match U-Net spatial levels
    c4  = F.interpolate(context['scale_4'],  size=(H//2, W//2), mode='bilinear', align_corners=False)
    c8  = F.interpolate(context['scale_8'],  size=(H//4, W//4), mode='bilinear', align_corners=False)
    c16 = F.interpolate(context['scale_16'], size=(H//8, W//8), mode='bilinear', align_corners=False)

    # Encoder
    e1 = self.enc1(x, t_emb)                                        # (B,  64, H,   W  )
    e2 = self.enc2(torch.cat([self.down1(e1), c4],  dim=1), t_emb)  # (B, 128, H/2, W/2)
    e3 = self.enc3(torch.cat([self.down2(e2), c8],  dim=1), t_emb)  # (B, 256, H/4, W/4)
    e4 = self.enc4(torch.cat([self.down3(e3), c16], dim=1), t_emb)  # (B, 512, H/8, W/8)

    # Bottleneck
    b = self.bottleneck(self.down4(e4), t_emb)  # (B, 512, H/16, W/16)

    # Decoder
    d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), t_emb)     # (B, 256, H/8, W/8)
    d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)     # (B, 128, H/4, W/4)
    d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)     # (B,  64, H/2, W/2)
    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)     # (B,  64, H,   W  )

    noise_pred = self.out_conv(F.silu(self.out_norm(d1)))            # (B,   3, H,   W  )
    return noise_pred


# 4. Gaussian Diffusion - Forward process and loss

class GaussianDiffusion(nn.Module):
  '''
    Handles the forward process and training loss.

    the U-Net is trained to predict the noise at each timestep
    at inference, noise is iterativley removed to preduce the clean image
  '''

  def __init__(self, unet, T=1000, beta_start=1e-4, beta_end=0.02, schedule='cos'):
    super().__init__()
    self.unet = unet
    self.T = T

    # how much noise to add at each timestep
    if schedule == 'cos':
      # Non liniear with cosaine
      s = 8e-3
      steps = torch.arange(T+1,dtype=torch.float32)
      f_t = torch.cos(((steps/T)+s)/(1+s)*(math.pi/2)) ** 2
      alphas_cumprod = (f_t/f_t[0]).clamp(1e-5,0.9999)
      betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(1e-5,0.9999)
      alphas = 1.0 - betas
      alphas_cumprod = alphas_cumprod[1:]
    else:
      # Linear noise schedule - how much noise to add at each timestep
      betas = torch.linspace(beta_start, beta_end, T)
      alphas = 1.0 - betas
      alphas_cumprod = torch.cumprod(alphas, dim=0) # cumulative product

    # Register as buffers so they move to GPU with .to(device)
    self.register_buffer('betas', betas)
    self.register_buffer('alphas', alphas)
    self.register_buffer('alphas_cumprod', alphas_cumprod)
    self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
    self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1-alphas_cumprod))

  def q_sample(self, x_0, t, noise=None):
    '''
    Forward process - add noise to clean image x_0 at timnestep t

    x_t = sqrt(a_t) * x_0 + sqrt(1 - a_t) * epsilon

    Input:
      x_0:    (B, 3, H, W)  clean image
      t:      (B,)          timestep per sample
      noise:  (B, 3, H, W)  optional - if None, sampled randomly
    Output:
      x_t:    (B, 3, H, W)  noisy image at timestep t
      noise:  (B, 3, H, W)  the noise that was added
    '''
    if noise is None:
      noise = torch.randn_like(x_0)

    sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_1_a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

    x_t = sqrt_a * x_0 + sqrt_1_a * noise
    return x_t, noise

  def training_loss(self, x_0, key_frame, context, lpips_vgg=None, t=None, deg_vec=None):
    '''
    Compute training loss - MSE between predicted and actual noise.

    Input:
      x_0:          (B, 3, H, W)  clean GT image
      key_frame:    (B, 3, H, W)  hazy key frame
      context:       dict         multi-scale conbtext from gating
      t:            (B,)          optional timestep, sampled randomly if None
    Output:
      loss:         scalar
      delta:        (B, 3, H, W)  predicted residual
    '''
    x_0 = 2*x_0 - 1

    # Sample random timesteps if not provided
    if t is None:
      t = torch.randint(0, self.T, (x_0.shape[0],), device=x_0.device)

    # Forward process - add noise to GT
    x_t, noise = self.q_sample(x_0, t)

    # U-Net predicts the noise
    noise_pred = self.unet(x_t, t, context, key_frame, deg_vec=deg_vec)

    # MSE loss between predicted and actual noise
    loss_diff = F.mse_loss(noise_pred, noise)

    # Compute residual delta for monitoring
    x0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

    # L1 loss
    loss_l1 = F.l1_loss(x0_pred, x_0)

    # LPIPS loss
    loss_lpips = lpips_vgg(x0_pred, x_0).mean()

    #combined loss
    loss = loss_diff + 0.1 * loss_l1 + loss_lpips*0.02

    with torch.no_grad():
        delta = x0_pred - (2 * key_frame - 1)

    return loss, delta

  def predict_x0_from_noise(self, x_t, t, noise_pred):
    '''
    reconstuct clean iamge estimnate from noisy image and predicted noise.

    x_0 = (x_t - sqrt(1 - a_t) * epsilon) / sqrt(a_t)
    '''

    sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None , None]
    sqrt_1_a= self.sqrt_one_minus_alphas_cumprod[t][:,None, None, None]
    sqrt_a = torch.clamp(sqrt_a, min=1e-4)

    x0_pred = (x_t - sqrt_1_a * noise_pred) / sqrt_a

    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
    return x0_pred

  @torch.no_grad()
  def sample(self, key_frame, context, T=None, deg_vec=None):
    '''
    Inference - iteratively denoise from pure noise to clean image
    final  output is key_frame + predicted residual.

    Input:
      key_frame:  (B, 3, H, W)  haze key frame
      context:    dict          multi-scale context
    Output:
      J:          (B, 3, H, W)  restored clean image
    '''
    T_full      = self.T
    T_steps           = T or T_full
    B, C, H, W  = key_frame.shape
    device      = key_frame.device

    # Pure noise — standard DDPM sampling
    #timesteps = list(reversed(torch.linspace(0,T_full-1, T_steps).long().tolist()))
    #x_t = torch.randn(B, 3, H, W, device=device)

    noise_strenght = 0.4
    start_t = max(1, int(noise_strenght * T_full) - 1)
    key_norm  = key_frame * 2 - 1
    t_start   = torch.full((B,), start_t, device = device, dtype=torch.long)
    noise     = torch.randn_like(key_norm)
    x_t       = self.q_sample(key_norm, t_start, noise=noise)[0]

    timesteps = torch.linspace(0,start_t, T_steps).long().flip(0).tolist()

    for i, t_val in enumerate(timesteps):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise_pred = self.unet(x_t, t, context, key_frame, deg_vec=deg_vec)

        # DDIM x0 estimate
        sqrt_a  = self.sqrt_alphas_cumprod[t_val]
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t_val]
        x0_pred = (x_t - sqrt_1a * noise_pred) / sqrt_a.clamp(min=1e-4)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if i < len(timesteps) - 1:
            t_prev       = timesteps[i + 1]
            sqrt_a_prev  = self.sqrt_alphas_cumprod[t_prev]
            sqrt_1a_prev = self.sqrt_one_minus_alphas_cumprod[t_prev]
            x_t = sqrt_a_prev * x0_pred + sqrt_1a_prev * noise_pred
        else:
            x_t = x0_pred  # final step

    x_t = x_t.clamp(-1.0, 1.0)
    J   = (x_t + 1) / 2
    return J


