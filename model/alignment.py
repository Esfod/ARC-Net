"""Optical flow with flow estinmation using RAFT"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.optical_flow as of

class FlowEstimator(nn.Module):
  '''
  Uses a pretrained RAFT-small to estimate optical flow between keyframe and each auxiliary frame


  Input:
    key_frame:  (B,3,H,W)
    aux_frame:  (B,3,H,W)
  Output:
    flow:       (B,2,H,W) - (dx,dy) displacement per pixel
  '''

  def __init__(self, freeze=True):
    super().__init__()

    #load pretrained RAFT-small
    self.raft = of.raft_small(weights=of.Raft_Small_Weights.DEFAULT)

    #freeze RAFT weights
    if freeze:
      for param in self.raft.parameters():
        param.requires_grad = False

  def forward(self, key_frame, aux_frame):
    #RAFT excpect images in [0, 255] range, we have tensor which is [0,1]
    key_255 = key_frame * 255.0
    aux_255 = aux_frame * 255.0

    #RAFT returns a list of flow predictions for each iterations
    # we take the last one which is most refined
    flow_predictions = self.raft(aux_255, key_255)
    flow = flow_predictions[-1]

    return flow

class FeatureWarper(nn.Module):
  '''
  Warps the Auxiliary features into key frame coordinate system
  using the estimated flow field.

  Input:
    feature:    (B, C, H, W)  -aux frame features at one scale
    flow:       (B, 2, H, W)  - flow at full resolution
  Output:
    warped:     (B, C, H, W)  - features aligned to key frame
  '''

  def forward(self, features, flow):
    B, C, H, W = features.shape

    #Check if flow and feature spatial is same size, if not resize
    if flow.shape[-2:] != features.shape[-2:]:
      scale_h = H / flow.shape[-2]
      scale_w = W / flow.shape[-1]
      flow = F.interpolate(flow, size=(H,W), mode='bilinear', align_corners=False)
      # Scale flow vectors to mathc the new resolution
      flow = flow * torch.tensor([scale_w, scale_h],
                                 device=flow.device).view(1,2,1,1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=features.device, dtype=torch.float32),
        torch.arange(W, device=features.device, dtype=torch.float32),
        indexing='ij'
    )
    grid = torch.stack([grid_x,grid_y], dim=0)      # (2, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1 , -1) # (B, 2, H, W)

    #Add flow to sampling location
    sampling_grid = grid + flow   # (B, 2, H, W)

    #Normalise to [-1, 1] as required by grid_sample
    sampling_grid[:,0] = 2.0 * sampling_grid[:,0] / (W - 1) - 1.0 # x
    sampling_grid[:,1] = 2.0 * sampling_grid[:,1] / (H - 1) - 1.0 # y

    sampling_grid = sampling_grid.permute(0, 2, 3, 1)

    #sample features at the new locations
    warped = F.grid_sample(
        features,
        sampling_grid,
        mode='bilinear',
        padding_mode='border', #uses border values for out-of bounds regions
        align_corners=True
    )
    return warped

class AlignmentModule(nn.Module):
  '''
  Alignsment Module - using flow and warps all
  auxiliary features into key frame coordinate system

  Inputs:
    key_frame:    (B, C, H, W)
    aux_frames:   (B, K, C, H, W)
    F_k:  list of K feature dicsts from encoder

  Output:
    aligned_features: list of K dicts with warped featuresa at each scale
    flows:            list of K flow tensors (B, 2, H, W)
  '''

  def __init__(self):
    super().__init__()
    self.flow_estimator = FlowEstimator()
    self.warper = FeatureWarper()

  def forward(self, key_frame, aux_frames, F_k):
    B, K, C, H, W = aux_frames.shape#(B, 3, H, W)
    aligned_features =  []
    flows = []

    for i in range(K):
      aux__frame = aux_frames[:,i] #(B, 2, H, W)

      #Estimate flow at full resolution
      flow = self.flow_estimator(key_frame,aux__frame)
      flows.append(flow)


      # Warp features  at each scale using the flow
      warped_dict = {}
      for scale, feat in F_k[i].items():
        warped_dict[scale] = self.warper(feat,flow)

      aligned_features.append(warped_dict)

    return aligned_features, flows