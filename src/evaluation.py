# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#
""" Evaluation Code
  We ported original MATLAB code for ISBI2015 into python.
"""

# Import global packages
import os
import sys
import math
import torch
from constants import *


def upscale_coordinates(img_batch, x, y):
  """Fix scale of annotation to the raw input (RAW_IMG_H, RAW_IMG_W).
  Args:  
    img_batch (torch.tensor): Output image to fix scale.
      Shape = [n_batch, n_ch, height, width]
    x (torch.tensor): X coordinates
      Shape = [n_batch, NUM_LM]
    y (torch.tensor): Y coordinates
      Shape = [n_batch, NUM_LM]
  
  Returns:
    up_x (torch.tensor): X coordinates
      Shape = [n_batch, NUM_LM]
    up_y (torch.tensor): Y coordinates
      Shape = [n_batch, NUM_LM]
  """
  _, _, img_h, img_w = img_batch.shape
  y_scale = RAW_IMG_H / img_h
  x_scale = RAW_IMG_W / img_w

  up_x = torch.round(x.float() * x_scale)
  up_y = torch.round(y.float() * y_scale)

  return up_x, up_y


def radial_error(img_batch, gt_x, gt_y, exp_x, exp_y):
  """ Return radial error.
  Calculate pixelwise Euclidean distance.

  Args:  
    img_batch (torch.tensor): Output image to fix scale.
      Shape = [n_batch, n_ch, height, width]
    gt_x (torch.tensor): Ground truth annotation (x coord) in RAW_IMG_W size.
      Shape = [n_batch, NUM_LM]
    gt_y (torch.tensor): Ground truth annotation (y coord) in RAW_IMG_H size.
      Shape = [n_batch, NUM_LM]
    exp_x (torch.tensor): Predicted x coordinate.
      Shape = [n_batch, NUM_LM]
    exp_y (torch.tensor): Predicted y coordinate.
      Shape = [n_batch, NUM_LM]

  Returns:
    rad_err (torch.tensor): Radial error for batches.
      Shape = [n_batch, NUM_LM]
  """
  up_exp_x, up_exp_y = upscale_coordinates(
    img_batch=img_batch, x=exp_x, y=exp_y
  )
  rad_err = torch.sqrt((up_exp_x - gt_x) ** 2 + (up_exp_y - gt_y) ** 2)
  
  return rad_err


def detection_with_bdd(rad_err, det_bdd_list=DET_BDD_LIST):
  """ Detection results with boundaries 20, 25, 30, 40.

  Args:  
    rad_err (torch.tensor): Radial error.
      Shape = [n_sample, NUM_LM]
    det_bdd_list (list): Theshold for SDR.
      Shape = [20, 25, 30, 40]

  Returns: 
    d_results (torch.tensor):
      Shape = [n_bdds, n_sample, NUM_LM]
  """
  detect_bdds = torch.tensor(det_bdd_list).reshape([len(det_bdd_list), 1, 1])

  return rad_err.repeat(len(det_bdd_list), 1, 1) <= detect_bdds


def stats_to_dict(tensor, stat_type, det_bdd_list=DET_BDD_LIST):
  """ Convert evaluation statistics into dictionary with LM names as keys.

  Args:  
    tensor (torch.tensor): Statitistics.
      Shape = [n_samples, NUM_LM] for 'MRE', 'SD'
            = [n_bdds, n_samples, NUM_LM] for 'SDR'
    stat_type (str): Type of statistics.
      Values = 'MRE'(mean radial error per each lm), 
               'SD' (mean stdev per each lm), 
               'SDR' (mean detection per each lm for each detection boundary),

  Returns: 
    stat_dict (dict):
      Shape = {'S':(torch.tensor), ..., 'average':(torch.tensor)} for 'MRE','SD'
      Shape = {'20': {'S': (torch.tensor[0]), ...},
               '25': {'S': (torch.tensor[0]), ...}, ...} for 'DET'
  """
  stat_dict = {}

  if stat_type == 'MRE':
    avg_tensor = tensor.mean(dim=0) # Average over samples.
    for lm_idx, val in enumerate(avg_tensor):
      stat_dict[S_LM_NAME_DICT[lm_idx]] = val
    stat_dict['average'] = tensor.mean()

  if stat_type == 'SD':
    avg_tensor = tensor.std(dim=0) # Average over samples.
    for lm_idx, val in enumerate(avg_tensor):
      stat_dict[S_LM_NAME_DICT[lm_idx]] = val
    stat_dict['average'] = avg_tensor.mean()

  if stat_type == 'SDR':
    avg_tensor = 100 * (tensor.mean(dim=1)) # Average over samples.
    for bdd_idx, bdd in enumerate(det_bdd_list):
      det_dict = {}
      for lm_idx, val in enumerate(avg_tensor[bdd_idx, :]):
        det_dict[S_LM_NAME_DICT[lm_idx]] = val
      det_dict['average'] = 100 * (tensor[bdd_idx, :].mean())
      stat_dict[bdd] = det_dict

  return stat_dict


def statistics(test_list, rst_dict, gt_dict):
  """ Evaluation statistics (MRE, SD, SDR, SCR)
  Args:  
    test_list (list): List of test names (str).
    rst_dict (dictionary): Test1, Test2 results.
    gt_dict (dictionary): Test1, Test2 ground truth labels.

  Returns:
    stat_dict (dictionary): Test1, Test2 statistics
      Shape = {'t1':{'MRE':mre}, {'SD':sd}, {'SDR':sdr}, {'SCR':scr},
               't2':{'MRE':mre}, {'SD':sd}, {'SDR':sdr}, {'SCR':scr}}
      mre (dictionary): Test1, Test2 MRE
        Shape = {'A':tensor, ... 'average':tensor}
      sd (dictionary): Test1, Test2 SD
        Shape = {'A':tensor, ... 'average':tensor}     
      sdr (dictionary): Test1, Test2 SDR
        Shape = {20:{'A':tensor, ...}, ... 40:{'A':tensor, ...}
      scr (tensor): Test1, Test2 SCR
        Shape = tensor([1, 8])
  """
  dx_facial_type = FacialTypeMeasurement()
  stat_dict = {}

  for t_name in test_list:

    rad_err = torch.tensor([]) # RadialError for each test

    rst = rst_dict[t_name]
    gt = gt_dict[t_name]
    gt_x_list = gt['x']
    gt_y_list = gt['y']
    rst_x_list = rst['x']
    rst_y_list = rst['y']
    
    for b_idx in range(gt['x'].shape[0]):

      # Get x y coordinates of ground truth and experiments results  
      gt_x = gt_x_list[b_idx]
      gt_y = gt_y_list[b_idx]
      rst_x = rst_x_list[b_idx]
      rst_y = rst_y_list[b_idx]

      # Calculate radial error
      rad_err_per_sample = radial_error(
        img_batch=torch.zeros([1, 1, RAW_IMG_H, RAW_IMG_W]),
        gt_x=gt_x, gt_y=gt_y, exp_x=rst_x, exp_y=rst_y).unsqueeze(0)
      rad_err = torch.cat([rad_err, rad_err_per_sample], dim=0)

    # Facial type measurements
    gt_ft = dx_facial_type.get_classes(
      img_batch=torch.zeros([1, 1, RAW_IMG_H, RAW_IMG_W]), 
      x=gt['x'], y=gt['y']
    ) 
    rst_ft = dx_facial_type.get_classes(
      img_batch=torch.zeros([1, 1, RAW_IMG_H, RAW_IMG_W]), 
      x=rst['x'], y=rst['y']
    )  
    rst_ft = rst_ft[:, 1, :]
    gt_ft = gt_ft[:, 1, :]

    # Calculate MRE, SDR, SD
    mre = stats_to_dict(rad_err, 'MRE')
    sd = stats_to_dict(rad_err, 'SD')
    sdr = stats_to_dict(detection_with_bdd(rad_err).float(), 'SDR')

    # Calculate SCR from diagoanl averages of facial types
    diag_ft_acc = 0
    for f_cl in range(1, 5):
      cl_rst = ((gt_ft == f_cl) & (rst_ft == f_cl)).float().sum(dim=0)
      cl_rst = cl_rst / ((gt_ft == f_cl)).float().sum(dim=0)
      cl_rst[torch.isnan(cl_rst)] = 0
      diag_ft_acc = diag_ft_acc + cl_rst
    scr = 100 * (diag_ft_acc / 3)
   
    # Store into dictionary
    stat_dict[t_name] = {
      'MRE':mre, 
      'SDR':sdr, 
      'SD':sd, 
      'SCR':scr
    }

  return stat_dict


class FacialTypeMeasurement:
  """ Wrapper class for 'angle.py' in ISBI2015 Evaluation Code in python3. """

  def __init__(self):
    pass

  def set_points_list(self, img_batch, x, y):
    """ Convert x, y coords into list of self.Point .
      Args: 
        img_batch (torch.tensor): Output image to fix scale.
          Shape: [n_batch, n_ch, height, width]
        x (torch.tensor): X coordinates 
          Shape: [n_batch, NUM_LM]
        y (torch.tensor): X coordinates 
          Shape: [n_batch, NUM_LM]
    """
    n_batch, _ = x.shape

    self.points_list = []
    self.n_batch = n_batch

    up_x, up_y = upscale_coordinates(img_batch=img_batch, x=x, y=y)

    for b_idx in range(n_batch):
      points = []
      for lm_idx in range(NUM_LM):
        points.append(
          self.Point(
            x=int(up_x[b_idx, lm_idx]), 
            y=int(up_y[b_idx, lm_idx]),
          )
        )
      self.points_list.append(points)


  class Point:
    def __init__(self, x, y):
      self.x = x
      self.y = y
    def __str__(self):
      return str(self.x) + "," + str(self.y)


  class Vector:
    def __init__(self, pa, pb):
      self.x = int(pb.x) - int(pa.x)
      self.y = int(pb.y) - int(pa.y)
    def __str__(self):
      return str(self.x) + "," + str(self.y)


  class Angle:
    def __init__(self, va, vb):
      self.va = va
      self.vb = vb
    def theta(self):
      theta = math.degrees(
        math.acos(
          (self.va.x * self.vb.x + self.va.y * self.vb.y) / \
          (math.hypot(self.va.x, self.va.y) * math.hypot(self.vb.x, self.vb.y))
        )
      )
      return theta


  class Distance:
    def __init__(self, pa, pb):
      self.x = (int(pb.x) - int(pa.x))*(int(pb.x) - int(pa.x))
      self.y = (int(pb.y) - int(pa.y))*(int(pb.y) - int(pa.y))
    def dist(self):
      return  (self.x+self.y)**0.5


  def get_cross_product(self, va, vb):
    return va.x*vb.y - va.y*vb.x
  

  def get_ANB(self, points):
    ANB = None
    ANB_type = None
    
    va = self.Vector(points[1], points[0])
    vb = self.Vector(points[1], points[5])
    vc = self.Vector(points[1], points[0])
    vd = self.Vector(points[1], points[4])
    ANB = self.Angle(vc, vd).theta() - self.Angle(va, vb).theta()
    
    if ANB < 3.2:
      ANB_type = 3
    elif ANB > 5.7:
      ANB_type = 2
    else:
      ANB_type = 1

    return ANB, ANB_type


  def get_SNB(self, points):
    SNB = None
    SNB_type = None
    va = self.Vector(points[1], points[0])
    vb = self.Vector(points[1], points[5])
    SNB = self.Angle(va, vb).theta()

    if SNB < 74.6:
      SNB_type = 2
    elif SNB > 78.7:
      SNB_type = 3
    else:
      SNB_type = 1

    return SNB, SNB_type


  def get_SNA(self, points):
    SNA = None
    SNA_type = None

    va = self.Vector(points[1],points[0])
    vb = self.Vector(points[1],points[4])
    SNA = self.Angle(va, vb).theta()

    if SNA < 79.4:
      SNA_type = 3
    elif SNA > 83.2:
      SNA_type = 2
    else:
      SNA_type = 1

    return SNA, SNA_type


  def get_ODI(self, points):  
    ODI = None
    ODI_type = None
    pa = points[7]
    pb = points[9]
    pc = points[5]
    pd = points[4]
    pe = points[3]
    pf = points[2]
    pg = points[16]
    ph = points[17]

    va = self.Vector(pa,pb)
    vb = self.Vector(pc,pd)
    vc = self.Vector(pe,pf)
    vd = self.Vector(pg,ph)

    aa = self.Angle(va,vb).theta()
    ab = self.Angle(vc,vd).theta()
    cb = self.get_cross_product(vc,vd)

    if cb < 0:
      ab = -ab
    
    ODI = aa+ab

    if ODI < 68.4:
      ODI_type = 3
    elif ODI > 80.5:
      ODI_type = 2
    else:
      ODI_type = 1

    return ODI, ODI_type



  def get_APDI(self, points):
    APDI = None
    APDI_type = None

    pa = points[2]
    pb = points[3]
    pc = points[1]
    pd = points[6]
    pe = points[4]
    pf = points[5]
    pg = points[3]
    ph = points[2]
    pi = points[16]
    pj = points[17]
   
    va = self.Vector(pa,pb)
    vb = self.Vector(pc,pd)
    vc = self.Vector(pe,pf)
    vd = self.Vector(pg,ph)
    ve = self.Vector(pi,pj)
    aa = self.Angle(va,vb).theta()
    ab = self.Angle(vb,vc).theta()
    ac = self.Angle(vd,ve).theta()
    cb = self.get_cross_product(vb,vc)
    cc = self.get_cross_product(vd,ve)

    if cb > 0:
      ab = -ab
    if cc < 0:
      ac = -ac

    APDI = aa+ab+ac

    if APDI < 77.6:
      APDI_type = 2
    elif APDI > 85.2:
      APDI_type = 3
    else:
      APDI_type = 1

    return APDI, APDI_type


  def get_FHI(self, points):
    FHI = None
    FHI_type = None
    pfh = self.Distance(points[0], points[9]).dist()
    afh = self.Distance(points[1], points[7]).dist()

    FHI = pfh / afh

    if FHI < 0.65:
      FHI_type = 3
    elif FHI > 0.75:
      FHI_type = 2
    else:
      FHI_type = 1

    return FHI, FHI_type

  def get_FMA(self, points):
    FMA = None
    FMA_type = None
    va = self.Vector(points[0], points[1])
    vb = self.Vector(points[9], points[8])

    FMA = self.Angle(va ,vb).theta()
    if FMA < 26.8:
      FMA_type = 3
    elif FMA > 31.4:
      FMA_type = 2
    else:
      FMA_type = 1
    
    return FMA, FMA_type


  def get_MW(self, points):
    MW = self.Distance(points[10], points[11]).dist() / 10
    MW_type = None
    if points[11].x < points[10].x:
      MW = -MW
    if MW >= 2:
      if MW <= 4.5:
        MW_type = 1
      else:
        MW_type = 4
    elif MW == 0:
      MW_type = 2
    else:
      MW_type = 3

    return MW, MW_type


  def get_classes(self, img_batch, x, y):
    """ Classification of face with 8 analysis methods.
      Args: 
        img_batch (torch.tensor): Output image to fix scale.
          Shape: [n_batch, n_ch, height, width]
        x (torch.tensor): X coordinates 
          Shape: [n_batch, NUM_LM]
        y (torch.tensor): X coordinates 
          Shape: [n_batch, NUM_LM]
      Returns:
        cl_face (torch.tensor): Value and class w.r.t 8 ananlysis methods.
          Shape: [n_batch, 2, NUM_METHODS]
    """
    self.set_points_list(img_batch=img_batch, x=x, y=y)

    cl_face = torch.zeros([self.n_batch, 2, NUM_METHODS])
    for b_idx in range(self.n_batch):
      points = self.points_list[b_idx]
      cl_face[b_idx, 0, 0], cl_face[b_idx, 1, 0]= self.get_ANB(points)
      cl_face[b_idx, 0, 1], cl_face[b_idx, 1, 1]= self.get_SNB(points)
      cl_face[b_idx, 0, 2], cl_face[b_idx, 1, 2]= self.get_SNA(points)
      cl_face[b_idx, 0, 3], cl_face[b_idx, 1, 3]= self.get_ODI(points)
      cl_face[b_idx, 0, 4], cl_face[b_idx, 1, 4]= self.get_APDI(points)
      cl_face[b_idx, 0, 5], cl_face[b_idx, 1, 5]= self.get_FHI(points)
      cl_face[b_idx, 0, 6], cl_face[b_idx, 1, 6]= self.get_FMA(points)
      cl_face[b_idx, 0, 7], cl_face[b_idx, 1, 7]= self.get_MW(points)

    return cl_face