from sys import _xoptions
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset


class PointLoader(Dataset):
  def __init__(self, root, remap_array, data_split="Train",
              device="cpu", max_size=40000):
    self.root = root
    self.velo_dir = os.path.join(root, "velodyne_ds")
    self.velo_files = sorted(os.listdir(self.velo_dir))
    self.split = data_split
    if not self.split == "Test":
      self.label_dir = os.path.join(root, "labels_ds")
      self.label_files = sorted(os.listdir(self.label_dir))
    self.device = device
    self.remap_array = remap_array
    self.max_size = max_size
  
  def __len__(self):
    return len(self.velo_files)

  def __getitem__(self, index):
    # Load point cloud
    velo_file = os.path.join(self.velo_dir, self.velo_files[index])
    # Fetch the point cloud
    
    pc = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
    pc = pc[:,0:3]  # Load point cloud data
   
    if self.split == "Train" and np.random.rand() > 0.5:
      pc[:, 0] = -pc[:, 0]

    if self.split == "Test":
      return pc, None
    # Load labels
    label_file = os.path.join(self.label_dir, self.label_files[index])
    #  Fetch labels from label_file
    label = np.fromfile(label_file, dtype=np.int32).reshape(-1) & 0xFFFF
  
    label = self.remap_array[label] #None
    # Downsample the points
    if self.split == "Train":
      indices = np.random.permutation(pc.shape[0])[:self.max_size]
      pc = pc[indices, :]
      label = label[indices]
    return pc, label

  def collate_fn(self, data):
    B = len(data)
    pc_numpy = [data[i][0] for i in range(B)]
    torch_pc = torch.tensor(np.array(pc_numpy), device=self.device)
    label_numpy = [data[i][1] for i in range(B)]
    torch_label = torch.tensor(np.array(label_numpy), device=self.device)
    return torch_pc, torch_label 


class PointNetEncoder(nn.Module):
  def __init__(self, cs, linear_out=None):
    super().__init__()

    self.net = torch.nn.Sequential()
    for i in range(1, len(cs)):
      #  Replace None with a linear layer from cs[i-1] to cs[i]
      self.net.add_module("Lin" + str(i), nn.Linear(cs[i-1], cs[i]))
      #Replace None with a batchnorm layer of size cs[i]
      self.net.add_module("Bn" + str(i), nn.BatchNorm1d(cs[i]))
       # Replace None with a ReLU Layer
      self.net.add_module("ReLU" + str(i), nn.ReLU())
    if linear_out is not None:
      # Replace None with a linear layer from cs[i] to linear_out
      self.net.add_module("LinFinal", nn.Linear(cs[i], linear_out))

  def forward(self, x):
    # Input x is a BxNxC matrix where N is number of points
    B, N, C = x.shape
   
    x = x.view(B*N, C) #None

    x = self.net.forward(x) #None
    # Reshape x into shape (B, N, C1) where C1 is the new channel dim
    x = x.view(B, N, x.shape[-1]) #None
    return x
    
# This module learns to combines global and local point features
class PointNetModule(nn.Module):
  def __init__(self, cs_en, cs_dec, num_classes=20):
    super().__init__()
    #  Create a PointNetEncoder with cs=cs_en
    self.enc = PointNetEncoder(cs_en) #None
 
    self.dec = PointNetEncoder(cs_dec, linear_out = num_classes) #None

  def forward(self, x):
    B, N, C = x.shape
    # Encoder
    #  Feed x through the PointNetEncoder
    point_feats = self.enc(x) #None
   
    global_feats = torch.max(point_feats,1)[0] #None
  
    C1 = global_feats.shape[-1]
    global_feats = global_feats.view(B,1,C1)
    global_feats = global_feats.repeat(1, N, 1) #None
    # Concatenate local and global features along channel dimension (2)
    joint_feats = torch.cat([point_feats, global_feats], dim=2) # None
    # Feed joint_feats through the decoder module
    out = self.dec(joint_feats) #None
    return out



    transformed_points = torch.bmm(x, point_trans) #None
   
    output_preds = self.joint_enc(transformed_points)#None
    return output_preds
    
# This module adds a T-Net
class PointNetFull(nn.Module):
  def __init__(self, cs_en, cs_dec, cs_t_en, cs_t_dec):
    super().__init__()
   
    self.t_enc = PointNetEncoder(cs_t_en) #None
 
    # Note that 9 comes from the product 3x3
    self.t_dec = PointNetEncoder(cs_t_dec, linear_out = 9) #None
 
    self.joint_enc = PointNetModule(cs_en, cs_dec) #None

  def forward(self, x):
    B, N, C = x.shape
    # T-Net
 
    t_feats = self.t_enc(x) # None
  
    t_feats = torch.max(t_feats, dim=1)[0] # None
 
    C1 = t_feats.shape[-1]
    t_feats = t_feats.unsqueeze(1) # None
   
    t_feats = self.t_dec(t_feats)# None
    t_feats = t_feats.view(-1,3,3)
    identity = torch.eye(3).unsqueeze(0).to(x.device)
    
    point_trans = identity + t_feats # None
    
 
    # torch.bmm() may be helpful
    transformed_points = torch.bmm(x, point_trans) #None
    # Joint Encoder

    output_preds = self.joint_enc(transformed_points)#None
    return output_preds


# Compute the per-class iou and miou
def IoU(targets, predictions, num_classes, ignore_index=0):
  intersections = torch.zeros(num_classes, device=targets.device)
  unions = torch.zeros_like(intersections)
  counts = torch.zeros_like(intersections)

  valid_mask = targets != ignore_index #None
  targets = targets[valid_mask]
  predictions = predictions[valid_mask]
  # Loop over classes and update the counts, unions, and intersections
  for c in range(num_classes):
    t_c = (targets == c)
    p_c = (predictions == c)
    intersections[c] = torch.sum(t_c & p_c).float()
    unions[c] = torch.sum(t_c | p_c).float()
    counts[c] = torch.sum(t_c).float()
  
   
    unions[c] = unions[c] + 0.00001

  iou = intersections / unions #None
  iou[counts == 0] = 1.0
  # Calculate mean, ignoring ignore index
  total_iou = torch.sum(iou[1:])
  total_classes = num_classes - 1 
  miou = total_iou / total_classes#None
  return iou, miou

    

