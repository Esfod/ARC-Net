import torch
import torch.nn as nn

from model.encoder import MultiFrameEncoder
from model.alignment import AlignmentModule
from model.diffusion import DiffusionUNet, GaussianDiffusion
from model.degradation_encoder import DegradationEncoder



class ARCNet(nn.Module):
  """
  Pipeline:
    1. DegradationEncoder   – frozen CNN; produces 736-dim conditioning vector
    2. MultiFrameEncoder    – Swin-T feature pyramid for key + aux frames
    3. AlignmentModule      – RAFT optical flow + feature warping
    4. Simple feature fusion – concat(F_0, mean(aligned aux)) → context
    5. GaussianDiffusion    – deg-vec-conditioned DDIM denoising
  """

  def __init__(self, crop_size=256, total_images=3, diffusion_steps=1000):
    super().__init__()
    self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.diffusion_steps = diffusion_steps

    self.degradation_encoder = DegradationEncoder().to(self.device)
    self.encoder    = MultiFrameEncoder(pretrained=True, freeze=False,
                                        crop_size=crop_size).to(self.device)
    self.alignment  = AlignmentModule().to(self.device)
    self.unet       = DiffusionUNet(embed_dim=256).to(self.device)
    self.diffusion  = GaussianDiffusion(self.unet, T=self.diffusion_steps,
                                        schedule='cos').to(self.device)

  def _build_context(self, F_0, aligned):
    """
    Simple fusion: concat key-frame features with the mean of aligned
    aux-frame features at each scale.  Produces 2C channels per scale,
    matching what the DiffusionUNet encoder levels expect.
    """
    return {
        scale: torch.cat(
            [F_0[scale],
             torch.stack([a[scale] for a in aligned]).mean(0)],
            dim=1,
        )
        for scale in F_0
    }

  def forward(self, key_frame, aux_frames, sample_steps=10):
    """
    Args:
        key_frame    (B, 3, H, W)
        aux_frames   (B, K, 3, H, W)
        sample_steps int — DDIM steps (0 → full diffusion_steps)

    Returns:
        restored  (B, 3, H, W)
    """
    if sample_steps == 0:
        sample_steps = self.diffusion_steps

    #  Degradation encoding (5-layer vector) 
    deg_vec = self.degradation_encoder(key_frame)    # (B, 736)

    #  Feature extraction & alignment 
    F_0, F_k       = self.encoder(key_frame, aux_frames)
    aligned, flows = self.alignment(key_frame, aux_frames, F_k)

    #  Simple context fusion 
    context = self._build_context(F_0, aligned)

    #  Diffusion sampling 
    restored = self.diffusion.sample(key_frame, context,
                                     T=sample_steps,
                                     deg_vec=deg_vec)
    return restored

  #  Checkpoint helpers 

  def save_checkpoint(self, path, optimizer=None, epoch=None,
                      score=0, psnr=0, ssim=0, lpips=0):
    torch.save({
        'epoch':      epoch,
        'psnr':       psnr,
        'ssim':       ssim,
        'lpips':      lpips,
        'score':      score,
        'degradation_encoder': self.degradation_encoder.state_dict(),
        'encoder':    self.encoder.state_dict(),
        'alignment':  self.alignment.state_dict(),
        'diffusion':  self.diffusion.state_dict(),
        'optimizer':  optimizer.state_dict() if optimizer is not None else None,
    }, path)

  def load_checkpoint(self, path, optimizer=None):
    checkpoint = torch.load(path, map_location=self.device)
    self.degradation_encoder.load_state_dict(checkpoint['degradation_encoder'])
    self.encoder.load_state_dict(checkpoint['encoder'])
    self.alignment.load_state_dict(checkpoint['alignment'])
    self.diffusion.load_state_dict(checkpoint['diffusion'])
    if optimizer is not None and checkpoint.get('optimizer') is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  PSNR:  {checkpoint['psnr']:.2f} dB")
    print(f"  SSIM:  {checkpoint['ssim']:.4f}")
    print(f"  LPIPS: {checkpoint['lpips']:.4f}")
    print(f"  Score: {checkpoint['score']:.4f}")
    return checkpoint['epoch']
