
import torch
import torch.nn as nn
import timm


class SwinFeaturePyramidEncoder(nn.Module):
  ''' Swin'T feature pyramid encoder for multi-frame input.  Extracts features at 3 scales:'''

  def __init__(self, pretrained=True, freeze=True,crop_size=256):
    super().__init__()

    #Load Swin-T with pretrained ImageNet weights

    self.swin = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        features_only=True,
        out_indices=(0,1,2),
        img_size=crop_size
    )
    '''
    # small
    self.swin = timm.create_model(
        'swin_small_patch4_window7_224',
        pretrained=pretrained,
        features_only=True,
        out_indices=(0,1,2),
        img_size=crop_size
    )'''
    #freeze all Swin weights, then unfreeze the last stage (layers_2) for fine-tuning
    for param in self.swin.parameters():
      param.requires_grad = False

    for param in self.swin.layers_2.parameters():
      param.requires_grad = True



    self.proj_4 = nn.Conv2d(96,96, kernel_size=1)
    self.proj_8 = nn.Conv2d(192,192, kernel_size=1)
    self.proj_16 = nn.Conv2d(384,384, kernel_size=1)

  def forward(self, x):
    features = self.swin(x)

    f_4 = features[0].permute(0,3,1,2)  # (B, 96, H/4, W/4)
    f_8 = features[1].permute(0,3,1,2)  # (B, 192, H/8, W/8)
    f_16 = features[2].permute(0,3,1,2) # (B, 384, H/16, W/16)

    return{
        'scale_4': self.proj_4(f_4),
        'scale_8': self.proj_8(f_8),
        'scale_16': self.proj_16(f_16),
    }


class MultiFrameEncoder(nn.Module):

  def __init__(self, pretrained=True, freeze=True, crop_size=256):
    super().__init__()
    self.encoder = SwinFeaturePyramidEncoder(pretrained=pretrained, freeze=freeze, crop_size=crop_size)

  def forward(self, key_frame, aux_frames):
    B, K, C, H, W = aux_frames.shape

    F_0 = self.encoder(key_frame)

    F_k = []
    for i in range(K):
      F_k.append(self.encoder(aux_frames[:,i]))

    return F_0, F_k