# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#
""" Global, Local, and Refinement Network Definitions """

# Import global packages
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

# Import Kornia library for soft argmax
from kornia.geometry.subpix import spatial_soft_argmax2d

# Import global constants
from constants import *


class TVSegModel(nn.Module):
  """ Torchvision segmentation models wrapper class"""

  def __init__(self, tv_model, aux_loss, in_ch, num_classes):
    """ We used 'deeplabv3', 'fcn' with resnet50 backbone in torchvision.models
    Args:
      tv_model (str): Name of torchvision segmentation model
      aux_loss (bool): Use auxillary loss (True) or not (False) for models
      in_ch (int): Number of channels of input 
      num_classes (int): Number of channels of network output
    """
    super(TVSegModel, self).__init__()

    self.aux_loss = aux_loss
    self.tv_model = torchvision.models.segmentation.__dict__[tv_model](
      num_classes=num_classes,
      aux_loss=self.aux_loss,
    )

    # Replace conv1 to accept in_ch size channels
    self.tv_model.backbone.conv1 = nn.Conv2d(in_ch, 
      64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )


  def forward(self, inputs):
    return self.tv_model(inputs)


  def get_train_params(self, lr):
    """ Return trainable parameters in the networks with corresponding learning
    rates.

    Args: 
      lr (float): Learning rate for parameters

    Returns:
      params_to_optimize (list): Parameters to be trained.
    """
    params_to_optimize = [
      {"params": 
        [p for p in self.tv_model.backbone.parameters() if p.requires_grad]},
      {"params": 
        [p for p in self.tv_model.classifier.parameters() if p.requires_grad]},
    ]
    if self.aux_loss:
      params = [
        p for p in self.tv_model.aux_classifier.parameters() if p.requires_grad
      ]
      params_to_optimize.append({"params": params, "lr": lr * 10})

    return params_to_optimize


  def calc_loss(self, img_batch, preds, gt_x, gt_y):
    """ Calculate MSE loss with P^G.
    1. Predict landmark positions with softargmax.
    2. Rescale predicted positions fit to input image size
    3. Calculate MSE loss between GT and predicted points.

    Args:
      img_batch (tensor): Input image used in the prediction.
      preds (tensor): Predicted P^Gs for landmakrs
      gt_x (tensor): GT x coordinates for landmarks
      gt_y (tensor): GT y coordinates for landmarks

    Returns:
      MSE loss (tensor).
      If the network use aux_loss, loss is computed the weighted sum of final 
      loss and aux_loss.
      If aus_loss = False, loss is the same as MSE loss of final prediction.
    """
    losses = {}
    _, _, img_h, img_w = img_batch.shape
    x_scale = RAW_IMG_W / img_w
    y_scale = RAW_IMG_H / img_h

    for name, x in preds.items(): # For aux_loss == True
      pred_xy = spatial_soft_argmax2d(x, normalized_coordinates=False)
      pred_x = x_scale * pred_xy[:, :, 0]
      pred_y = y_scale * pred_xy[:, :, 1]
      pt_loss = F.mse_loss(pred_x, gt_x) + F.mse_loss(pred_y, gt_y)
      losses[name] = pt_loss
  
    if len(losses) == 1: # For aux_loss == False
      return losses['out']
  
    return losses['out'] + 0.5 * losses['aux']


  def calc_pat_loss(self, preds, gt_x, gt_y):
    """ Calculate MSE loss with P^L.
    1. Predict landmark positions with softargmax.
    2. Calculate MSE loss between GT and predicted points.

    Args:
      preds (tensor): Predicted P^Ls for landmakrs
      gt_x (tensor): GT x coordinates for landmarks
      gt_y (tensor): GT y coordinates for landmarks

    Returns:
      MSE loss (tensor).
      If the network use aux_loss, loss is computed the weighted sum of final 
      loss and aux_loss.
      If aus_loss = False, loss is the same as MSE loss of final prediction.
    """
    losses = {}

    for name, x in preds.items():
      pred_xy = spatial_soft_argmax2d(x, normalized_coordinates=False)
      pred_x = pred_xy[:, :, 0]
      pred_y = pred_xy[:, :, 1]
      pt_loss = F.mse_loss(pred_x, gt_x) + F.mse_loss(pred_y, gt_y)
      losses[name] = pt_loss
  
    if len(losses) == 1:
      return losses['out']
  
    return losses['out'] + 0.5 * losses['aux']


class RefineModel(nn.Module):
  """ Linear filter for refinement stage.
  Use nn.Linear without ReLU and bias as a linear filter.
  """

  def __init__(self, bilateral_lm_idx):
    """ Give bilateral landmark index to be targeted.

    Args:
      bilateral_lm_idx (int): Index of landmarks to be predicted with linear 
                              filter.
    """
    super(RefineModel, self).__init__()
    self.linear = nn.Linear(
      in_features=19*4, 
      out_features=2, 
      bias=False
    )

    # Initialize weights to put initial output is near the 
    # average of global and local stages.
    init_weight = torch.zeros_like(self.linear.weight.data)
    for i, lm_idx in enumerate(bilateral_lm_idx):
      init_weight[i, [lm_idx, 19 + lm_idx]] = 0.5
      init_weight[i + len(bilateral_lm_idx), [38 + lm_idx, 57 + lm_idx]] = 0.5   
    
    init_weight = init_weight +  0.001 * torch.rand_like(init_weight)
    
    self.linear.weight.data = init_weight


  def forward(self, x):
    return self.linear(x)


  def get_train_params(self, lr):
    """ Return trainable parameters in the networks with corresponding learning
    rates.

    Args: 
      lr (float): Learning rate for parameters

    Returns:
      params_to_optimize (list): Parameters to be trained.
    """    
    params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
    
    return params_to_optimize


  def calc_loss(self, preds, gt):
    """ Calculate MSE loss with regularization.
    The regulaization on weight of linear filter makes predictions as the 
    interanl divisions of predicted landmark positions from global and local 
    stages.

    Args:
      preds (tensor): Predicted landmark positions
      gt (tensor): GT for landmarks

    Returns:
      weighted sum of regularization term and MSE loss between predictions and
      ground gruth.
    """

    # Regularization on weight. Make predicted points as the interanal division
    # of predictions from global and local stages.
    weight_loss = F.mse_loss( 
      self.linear.weight.sum(dim=1), 
      torch.ones_like(self.linear.weight.sum(dim=1)).to('cuda')
    )

    # Pointwise MSE loss
    point_loss = F.mse_loss(preds, gt)

    return weight_loss + point_loss