# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#
""" Dataset for ISBI_2015"""

# Import global packages
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import cv2
from matplotlib import pyplot as plt

# Kornia library for data augmentation
from kornia import augmentation as K
import kornia.augmentation.functional as KF
import kornia.augmentation.random_generator as KRG

# Import local functions
from evaluation import upscale_coordinates

# Import global constants
from constants import *


class ISBIDataSet(object):
  """ Read ISBI2015 data and return images and labels.
    Format is:
     image (torch.tensor), 
     label(dictionary): {'ans_x': ANnotation of Senor X coordinate},
       {'ans_y': ANnotation of Senor Y coordinate},
       {'ans_c': ANnotation of Senor Classification},
       {'anj_x': ANnotation of Junior X coordinate},
       {'anj_y': ANnotation of Junior Y coordinate},
       {'anj_c': ANnotation of Junior Classification}
    
    Note:
      1. We used the average of 'ans' and 'anj' as ground truth
      2. Thus, the ground truth of facial classification is calculated from
         evaluation of 'ana' not from annotation files.
  """
  def __init__(self, data_root, mode, img_h, img_w, transforms, y_ch=False):
    """ Transforms and downsampling are determined with 'transforms'
    If transforms=ToTensor(), image is not downsampled and 'img_h'
    and 'img_w' be obsolete.
    If transforms=None, image is donwsampled as ('img_h', 'img_w')
    
    Args:
      data_root(str): Path of ISBI2015 dataset.
      mode(str): Dataset mode in [train, test1, test2].
      img_h(int): Height of image (used for downsampling)
      img_w(int): Width of image (used for downsampling)
      transforms(torchvision.transforms): Transforms to be applied. If it is 
        'None', then torchvision.transforms.ToTensor() is applied.
      y_ch(bool): Use Y channel image as input (True) image or RGB (False).
    """
    if mode == 'train':
      self.data_prefix = "TrainingData"
    elif mode == 'test1':
      self.data_prefix = "Test1Data"
    elif mode == 'test2':
      self.data_prefix = "Test2Data"
    else:
      assert('Error in mode')
    
    self.img_size = (img_h, img_w)
    self.img_scale = (img_h / RAW_IMG_H, img_w / RAW_IMG_W)
    self.transforms = transforms
    self.y_ch = y_ch

    if transforms is not None:
      self.transforms = transforms
    else:
      self.transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(self.img_size),
        torchvision.transforms.ToTensor(),]
      )

    self.data_root = data_root
    self.img_root = os.path.join(
      os.path.join(self.data_root, "RawImage"),
      self.data_prefix
    )
    self.ans_root = os.path.join(
      os.path.join(self.data_root, "AnnotationsByMD/senior"),
      self.data_prefix
    )
    self.anj_root = os.path.join(
      os.path.join(self.data_root, "AnnotationsByMD/junior"),
      self.data_prefix
    )

    self.img_list = list(sorted(os.listdir(self.img_root)))
    self.ans_list = list(sorted(os.listdir(self.ans_root)))
    self.anj_list = list(sorted(os.listdir(self.anj_root)))

  def __getitem__(self, idx):
    """ We used the average of 'ans' and 'anj' as ground truth ('ana') and 
    to fit to the scale, we also calculate 'ana_fs' that indicate the 'ana' in
    the down sampled images.

    The shape of ground-truth data is
      ann = {
        'ans_x': Annotation of x coordinate by senior in text file 
        'ans_y': Annotation of y coordinate by senior in text file 
        'anj_x': Annotation of x coordinate by junior in text file 
        'anj_y': Annotation of x coordinate by junior in text file 
        'ana_x': Average of 'ans_x' and 'anj_x'
        'ana_y': Average of 'ans_y' and 'anj_y'
        'ans_x_fs': Scaled 'ans_x' for down sampled input image 
        'ans_y_fs': Scaled 'ans_y' for down sampled input image 
        'anj_x_fs': Scaled 'anj_x' for down sampled input image 
        'anj_y_fs': Scaled 'anj_y' for down sampled input image 
        'ana_x_fs': Scaled 'ana_x' for down sampled input image
        'ana_y_fs': Scaled 'ana_y' for down sampled input image
        'ans_c': Annotation of facial class type by senior in text file
        'anj_c': Annotation of facial class type by junior in text file 
        'ana_c': (deprecated) Set as the same as 'ans_c'
      }
    """
    # load images ad masks
    img_path = os.path.join(self.img_root, self.img_list[idx])
    ans_path = os.path.join(self.ans_root, self.ans_list[idx])
    anj_path = os.path.join(self.anj_root, self.anj_list[idx])

    pil_img = Image.open(img_path).convert("RGB")
    img = self.transforms(pil_img) # Load image
    with open(ans_path) as ans_f: # Read lines without '\n'
      ans = [ans_l.rstrip() for ans_l in ans_f]
    with open(anj_path) as anj_f: # Read lines without '\n'
      anj = [anj_l.rstrip() for anj_l in anj_f]

    # Annotation
    ann = {} 
    
    # Annotation by Senior. (_fs means 'fixed scale')
    ann["ans_x"] = np.array([(float(xy.split(',')[0])) for xy in ans[:NUM_LM]])
    ann["ans_y"] = np.array([(float(xy.split(',')[1])) for xy in ans[:NUM_LM]])
    ann["ans_x_fs"] = self.img_scale[1] * ann["ans_x"]
    ann["ans_y_fs"] = self.img_scale[0] * ann["ans_y"]
    
    # Annontation by Junior.
    ann["anj_x"] = np.array([(float(xy.split(',')[0])) for xy in anj[:NUM_LM]])
    ann["anj_y"] = np.array([(float(xy.split(',')[1])) for xy in anj[:NUM_LM]])
    ann["anj_x_fs"] = self.img_scale[1] * ann["anj_x"]
    ann["anj_y_fs"] = self.img_scale[0] * ann["anj_y"]
    
    # Averaged annotation.
    ann["ana_x"] = 0.5 * (ann["ans_x"] + ann["anj_x"])
    ann["ana_y"] = 0.5 * (ann["ans_y"] + ann["anj_y"])
    ann["ana_x_fs"] = 0.5 * (ann["ans_x_fs"] + ann["anj_x_fs"])
    ann["ana_y_fs"] = 0.5 * (ann["ans_y_fs"] + ann["anj_y_fs"])
    
    # Face type   
    ann["ans_c"] = np.pad(np.array([int(c) for c in ans[NUM_LM:]]), (0, 11))
    ann["anj_c"] = np.pad(np.array([int(c) for c in anj[NUM_LM:]]), (0, 11))   
    ann["ana_c"] = ann["ans_c"]

    if self.y_ch == False:
      return img, ann
    else:
      y_ch_img =  self.transforms(pil_img.convert("YCbCr").getchannel('Y'))
      return img, ann, y_ch_img

  def __len__(self):
    return len(self.img_list)


def to_numpy_image(tensor_img):
  return tensor_img.transpose(1, 3).transpose(1, 2).cpu().numpy()


def to_tensor_image(np_img):
  return torch.tensor(np.transpose(np_img, (0, 3, 1, 2)))


def to_numpy_arr(tensor_arr):
  return tensor_arr.cpu().numpy()


def to_tensor_arr(np_arr):
  return torch.tensor(np_arr)


def vis_isbi(img_batch, pred_batch, x, y, c, radius, font_scale, txt_offset):
  """ Visualize predicted (or ground truth) landmark positions as circle
  in the input images.

  Args:
    img_batch (torch.tensor): Raw input image from ISBI2015
    pred_batch (torch.tensor): Image used for the prediction (e.g. down sampled)
    x (torch.tensor): (Predicted) landmark positions (x coordinate) 
    y (torch.tensor): (Predicted) landmark positions (y coordinate) 
    c (torch.tensor): (Deprecated) (predicted) facial class type
    radius (int): Radius of circle of landmark 
    font_scale (int): Size of landmark text (short names)
    txt_offset (int): Offset distance of text from landmark locations

  Returns:
    vis_img (tensor): Result image
  """  
  n_batch, img_c, img_h, img_w = img_batch.shape
  _, pred_c, pred_h, pred_w = pred_batch.shape

  x = ((img_w / pred_w) * to_numpy_arr(x)).astype(np.int)
  y = ((img_h / pred_h) * to_numpy_arr(y)).astype(np.int)

  num_lm = x.shape[1]
  img_batch = to_numpy_image(img_batch)
  vis_img = np.zeros_like(img_batch)

  for n in range(n_batch):
  
    img = cv2.UMat(img_batch[n])
  
    for i in range(num_lm):
      img = cv2.circle(img=img, 
        center=(x[n, i], y[n, i]), 
        radius=radius, 
        color=(1, 0, 0), 
        thickness=-1,
      )
      img = cv2.putText(img=img, 
        text='{}'.format(S_LM_NAME_DICT[i]), 
        org=(x[n, i] + txt_offset, y[n, i] + txt_offset),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=font_scale, 
        color=(0, 1, 0), 
        thickness=2, 
        lineType=cv2.LINE_AA
      )
    overlayed_img = np.array(img.get())
  
    if len(overlayed_img.shape) == 2: # For gray scale image
      vis_img[n,:,:,0] = np.array(img.get())
    else:
      vis_img[n,:,:,:] = np.array(img.get())

  return to_tensor_image(vis_img)


def ann_to_heatmap(img_batch, ksize, sigma, x, y, c):
  """ Convert annotation into heatmaps of landmark locations using Gaussian
  distribution

  Args:
    img_batch (torch.tensor): Input image
    ksize (int): Size of Gaussian kernel (2 * ksize + 1)
    sigma (int): Sigma of Gaussian kernel
    x (torch.tensor): Landmark positions (x coordinate)
    y (torch.tensor): Landmark positions (y coordinate)
    c (torch.tensor): (Deprecated) Facial type
  
  Returns:
    gt_heatmap (tensor): Heatmatp of ground truth
  """
  n_batch, _, img_h, img_w = img_batch.shape
  n_lm = x.shape[1]
  
  x = torch.round(x).int()
  y = torch.round(y).int()

  g_mask = cv2.getGaussianKernel(2 * ksize + 1, sigma)
  g_mask = g_mask * g_mask.transpose()
  g_mask = torch.tensor(g_mask / np.max(g_mask))

  gt_heatmap = torch.zeros([n_batch, n_lm, img_h, img_w])

  for n in range(n_batch):
    for i in range(n_lm):
      gt_heatmap[n, i, y[n, i], x[n, i]] = 1

  return gt_heatmap


def heatmap_to_ann(heatmap_batch):
  """ Convert heatmap into series of X,Y coordinate by applying argmax.
  Args:
    heatmap_batch (torch.tensor)

  Returns: Integer coordinates (x, y)
  """
  n_batch, n_lm, img_w, img_h = heatmap_batch.shape
  x = torch.zeros([n_batch, n_lm])
  y = torch.zeros([n_batch, n_lm])

  for n in range(n_batch):
    for i in range(n_lm):
      raw_idx = heatmap_batch[n, i, :, :].argmax()
      y[n, i] = raw_idx // img_h
      x[n, i] = raw_idx - (y[n, i] * img_h)
  
  return x.int(), y.int()


def augmentation(
  img_batch, 
  heatmap_batch, 
  x, 
  y, 
  degrees, 
  scale, 
  brightness, 
  contrst, 
  saturation, 
  hue,
  same_on_batch):
  """ Augment cephalogram and heatmap with following step.
  1. Rotation: Use image center or porion as ceter of rotation.
  2. Scaling: Use image center or porion as ceter of rotation.
  3. Color jittering: Perturb brightness, contrast, stauration and hue.
  
  Args:
    img_batch (torch.tensor): Cephalogram from ISBI2015.
      Shape = [n_batch, n_ch, height, width]
    heatmap_batch (torch.tensor): GT heatmap.
      Shape = [n_batch, n_ch, height, width]
    x (torch.tensor): X coordinates of landmarks
      Shape = [n_batch, NUM_LM]
    y (torch.tensor): Y coordinates of landmarks
      Shape = [n_batch, NUM_LM]
    degrees (list): Range of random rotation.
      Shape = [int, int]
    scale (int): Range of random scale.
    brightness (int): Range of random brightness.
    contrst (int): Range of random contrast.
    stauration (int): Range of random stauration.
    hue (int): Range of random hue.
    same_on_batch(bool): Same on batch.

  Returns:
    aug_img (torch.tensor): Augmented cephalograms.
      Shape = [n_batch, n_ch, height, width]
    aug_heatmap (torch.tensor): Augmented heatmaps.
      Shape = [n_batch, n_ch, height, width]
    aug_x (torch.tensor): X coordinates of augmented cephalograms' landmarks
      scaled as ISBI2015 
      Shape = [n_batch, NUM_LM]
    aug_y (torch.tensor): Y coordinates of augmented cephalograms' landmarks
      scaled as ISBI2015 
      Shape = [n_batch, NUM_LM]
    aug_x_fs (torch.tensor): X coordinates of augmented cephalograms' landmarks
      scaled as heatmap
      Shape = [n_batch, NUM_LM]
    aug_y_fs (torch.tensor): Y coordinates of augmented cephalograms' landmarks
      scaled as heatmap
      Shape = [n_batch, NUM_LM]
  """  
  n_batch, img_c, img_h, img_w = img_batch.shape
  aff_degrees = degrees
  aff_scale = scale

  affine_params = KRG.random_affine_generator(
    batch_size=n_batch, 
    height=img_h, 
    width=img_w, 
    degrees=aff_degrees, 
    scale=aff_scale,
    same_on_batch=same_on_batch,
  )

  color_jitter_params = KRG.random_color_jitter_generator(
    batch_size=n_batch, 
    brightness=brightness, 
    contrast=contrst, 
    saturation=saturation, 
    hue=hue, 
    same_on_batch=same_on_batch)
  
  aug_imgs = KF.apply_affine(img_batch, affine_params)
  aug_heatmaps = KF.apply_affine(heatmap_batch, affine_params)
  aug_x_fs, aug_y_fs = heatmap_to_ann(aug_heatmaps)
  aug_x, aug_y = upscale_coordinates(
    img_batch=img_batch, x=aug_x_fs, y=aug_y_fs
  )

  return aug_imgs, aug_heatmaps, aug_x_fs, aug_y_fs, aug_x, aug_y


def crop_lm_patches(img_batch, x_c_batch, y_c_batch, ann_batch, pat_sz):
  """ Cropping patches for local stage
  
  Args:
    img_batch (tensor): Input image
    x_c_batch (tensor): Crop center 'x'
    y_c_batch (tensor): Crop center 'y'
    ann_batch (tensor): Ground truth annotation
    pat_sz (int): Side length of patch

  Returns:
    img_crop_batch_list (tensor): Cropped patch images
    ana_x_batch_list (tensor): Landmark coordinates 'x' of patches
    ana_y_batch_list (tensor): Landmark coordinates 'y' of patches
  """
  img_crop_batch_list = []
  ana_x_batch_list = []
  ana_y_batch_list = []

  # Zero padding for cropping
  img_batch = F.pad(img_batch, (pat_sz, pat_sz, pat_sz, pat_sz))

  for img_idx in range(img_batch.shape[0]):
  
    img_crop_ch_list = []
    ana_x_ch_list = []
    ana_y_ch_list = []    
    
    # Padding requires offset GT and crop center by pat_sz.
    ana_x = int(ann_batch['ana_x'][img_idx]) + pat_sz
    ana_y = int(ann_batch['ana_y'][img_idx]) + pat_sz
    x_c = int(x_c_batch[img_idx]) + pat_sz
    y_c = int(y_c_batch[img_idx]) + pat_sz
    
    # ROI of patch
    pat_x_r = slice(x_c - pat_sz, x_c + pat_sz)
    pat_y_r = slice(y_c - pat_sz, y_c + pat_sz)

    # Cropped image
    img_crop = img_batch[img_idx:img_idx + 1, :, pat_y_r, pat_x_r].clone()
    img_crop_ch_list.append(img_crop) 

    # Annotation of patch is 
    # GT landmark position - crop center + patch_size
    ana_x_ch_list.append(torch.tensor([[pat_sz + ana_x - x_c]]))
    ana_y_ch_list.append(torch.tensor([[pat_sz + ana_y - y_c]]))

    img_crop_batch_list.append(torch.cat(img_crop_ch_list, dim=1))
    ana_x_batch_list.append(torch.cat(ana_x_ch_list, dim=1))
    ana_y_batch_list.append(torch.cat(ana_y_ch_list, dim=1))

  img_crop_batch_list = torch.cat(img_crop_batch_list, dim=0)
  ana_x_batch_list = torch.cat(ana_x_batch_list, dim=0)
  ana_y_batch_list = torch.cat(ana_y_batch_list, dim=0)    
  
  return img_crop_batch_list, ana_x_batch_list, ana_y_batch_list


def vis_patch(img_batch, x, y, c, radius, font_scale, txt_offset, lm_idx):
  """ Visualize predicted (or ground truth) landmark positions as circle
  in the cropped patches.

  Args:
    img_batch (torch.tensor): Cropped patch image
    x (torch.tensor): (Predicted) landmark positions (x coordinate) 
    y (torch.tensor): (Predicted) landmark positions (y coordinate) 
    c (torch.tensor): (Deprecated) (predicted) facial class type
    radius (int): Radius of circle of landmark 
    font_scale (int): Size of landmark text (short names)
    txt_offset (int): Offset distance of text from landmark locations
    lm_idx (int): Index of landmark to visualize

  Returns:
    vis_img (tensor): Result image
  """  
  n_batch, img_c, img_h, img_w = img_batch.shape
  x = to_numpy_arr(x).astype(np.int)
  y = to_numpy_arr(y).astype(np.int)

  num_lm = x.shape[1]
  img_batch = to_numpy_image(img_batch)
  vis_img = np.zeros_like(img_batch)
  for n in range(n_batch):
    img = cv2.UMat(img_batch[n])
    img = cv2.circle(img=img, 
      center=(x[n], y[n]), 
      radius=radius, 
      color=(1, 0, 0), 
      thickness=-1,
    )
    img = cv2.putText(img=img, 
      text='{}'.format(S_LM_NAME_DICT[lm_idx]), 
      org=(x[n] + txt_offset, y[n] + txt_offset),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
      fontScale=font_scale, 
      color=(0, 1, 0), 
      thickness=2, 
      lineType=cv2.LINE_AA
    )
    overlayed_img = np.array(img.get())
    if len(overlayed_img.shape) == 2:
      vis_img[n,:,:,0] = np.array(img.get())
    else:
      vis_img[n,:,:,:] = np.array(img.get())

  return to_tensor_image(vis_img)