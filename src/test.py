# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#

# Import global packages
import os
import copy
import argparse
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import SequentialSampler, DataLoader

# Import global constants
from constants import *

# Import user defined classes and functions
from isbi_dataset import crop_lm_patches
from evaluation import statistics
from datetime import datetime
from kornia.geometry.subpix import spatial_soft_argmax2d
from models import TVSegModel, RefineModel
from utils import write_result, save_as_mat, print_stat, get_loader_dict


def global_stage(
  gs_model_name, 
  gs_model_path,
  gs_aux_loss, 
  gs_img_h, 
  gs_img_w, 
  device, 
  test_list, 
  loader_dict):
  """Global stage (calculate P^G and x^G).
  
  Args:  
    gs_model_name (str): Name of torchvision.models for global stage to be used
    gs_model_path (str): Path of trained model for global stage (.pth file)
    gs_aux_loss (bool): Use of aux_loss for global stage (default is True)
    gs_img_h (int): Size of downsampled image (height)
    gs_img_w (int): Size of downsampled image (width)
    test_list (list): List of test names (str)
    loader_dict (dict): Dictionary of data_loader (key:test_name)
    device (str): Cuda or CPU  
  
  Returns:
    gs_dict (dictionary): Test1, Test2 results from global stage
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}
    gt_dict (dictionary): Test1, Test2 ground truth labels
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}
  """
  # Return dictionaries
  gs_dict = {} # Global stage result
  gt_dict = {} # Ground truth 

  # Load global stage model
  gs_model = TVSegModel(
    tv_model=gs_model_name, 
    aux_loss=gs_aux_loss, 
    in_ch=3, 
    num_classes=19
  )
  gs_model.to(device)
  gs_ckpt = torch.load(gs_model_path)
  gs_model.load_state_dict(gs_ckpt['model'])
  gs_model.eval()
  
  with torch.no_grad():
    for t_name in test_list:
      
      t_loader = loader_dict[t_name]
      gs_dict[t_name] = {'x':[], 'y':[]}
      gt_dict[t_name] = {'x':[], 'y':[]}
      
      for t_batch in t_loader:
        
        # Predict P^G
        t_img, t_ann = t_batch[0], t_batch[1]
        t_img = t_img.float().to(device)    
        t_out = gs_model(t_img)

        # Predict x^G with softargmax (expectation)
        t_xy = spatial_soft_argmax2d( 
          t_out['out'], normalized_coordinates=False
        )
        
        # Correct scales 
        t_x = (RAW_IMG_W / gs_img_w) * t_xy[:, :, 0].cpu()
        t_y = (RAW_IMG_H / gs_img_h) * t_xy[:, :, 1].cpu()

        # Save results
        gs_dict[t_name]['x'].append(t_x)
        gs_dict[t_name]['y'].append(t_y)
        gt_dict[t_name]['x'].append(t_ann['ana_x'])
        gt_dict[t_name]['y'].append(t_ann['ana_y'])

      # Reshape data
      gs_dict[t_name]['x'] = torch.cat(gs_dict[t_name]['x'], dim=0) 
      gs_dict[t_name]['y'] = torch.cat(gs_dict[t_name]['y'], dim=0) 
      gt_dict[t_name]['x'] = torch.cat(gt_dict[t_name]['x'], dim=0) 
      gt_dict[t_name]['y'] = torch.cat(gt_dict[t_name]['y'], dim=0) 

  # Free cuda memory 
  del(gs_model)
  
  return gs_dict, gt_dict


def local_stage(
  ls_model_name, 
  ls_model_path,
  ls_aux_loss, 
  pat_sz,
  test_list, 
  gs_dict, 
  loader_dict, 
  device):

  """Local stage (calculate P^L and x^L).
  Args:  
    ls_model_name (str): Name of torchvision.models for local stage to be used
    ls_model_path (str): Path of trained model for local stage (.pth file)
    ls_aux_loss (bool): Use of aux_loss for local stage (default is True)
    pat_sz (int): Side length of patch image
    test_list (list): List of test names (str)
    loader_dict (dict): Dictionary of data_loader (key:test_name)
    gs_dict (dictionary): Result dictionary from global stage
    device (str): Cuda or CPU  
  
  Returns:
    ls_dict (dictionary): Test1, Test2 results from local stage
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}
  """
  # Return dictionary
  ls_dict = copy.deepcopy(gs_dict)

  # Loop over landmarks
  for lm_idx in range(NUM_LM):  
  #for lm_idx in range(1):
    
    print('| Proc. {}th landmark {}'.format(lm_idx + 1, S_LM_NAME_DICT[lm_idx]))
    
    # Load local stage model for one landmark
    ls_model = TVSegModel(
      tv_model=ls_model_name, 
      aux_loss=ls_aux_loss, 
      in_ch=3, 
      num_classes=1
    )
    ls_model.to(device)
    ls_ckpt = torch.load(ls_model_path.format(
      S_LM_NAME_DICT[lm_idx]), 
      map_location='cpu'
    )
    ls_model.load_state_dict(ls_ckpt['model'])
    ls_model.eval()
      
    with torch.no_grad():
      for t_name in test_list:

        gs_rst = gs_dict[t_name]
        ls_rst = ls_dict[t_name]
        loader = loader_dict[t_name]

        for b_idx, (raw_img, ann) in enumerate(loader):
          
          # Global stage results and ground truth
          gs_x = gs_rst['x'][b_idx][lm_idx:lm_idx+1]
          gs_y = gs_rst['y'][b_idx][lm_idx:lm_idx+1]
          ann = {k:v[:, lm_idx] for k, v in ann.items()}

          # Predict P^L
          pat_img, pat_ana_x, pat_ana_y = crop_lm_patches(
            img_batch=raw_img.float(), 
            ann_batch=ann, 
            pat_sz=pat_sz, 
            x_c_batch=gs_x,
            y_c_batch=gs_y,
          )
          ls_img = pat_img.to(device)
          ls_xy = ls_model(ls_img)        

          # Predict x^L with softargmax (expectation)
          ls_xy = spatial_soft_argmax2d(
            ls_xy['out'], 
            normalized_coordinates=False
          )

          # Correct scales 
          ls_x = ls_xy[:, :, 0].cpu() + gs_x - pat_sz
          ls_y = ls_xy[:, :, 1].cpu() + gs_y - pat_sz

          # Save results
          ls_rst['x'][b_idx][lm_idx] = ls_x
          ls_rst['y'][b_idx][lm_idx] = ls_y

    # Free cuda memory 
    del(ls_model)

  return ls_dict


def refine_stage(rs_model_path, test_list, gs_dict, ls_dict, device):

  """Refinement stage (calculate x^P).
  Args:  
    rs_model_path (str): Path of trained model for refinement stage (.pth file)
    test_list (list): List of test names (str)
    gs_dict (dictionary): Result dictionary from global stage
    ls_dict (dictionary): Result dictionary from local stage
    device (str): Cuda or CPU  
  
  Returns:
    rs_dict (dictionary): Test1, Test2 results from refinement stage
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}
  """  
  # Return dictionary
  rs_dict = copy.deepcopy(ls_dict)

  # Load refinement stage models
  rs_ar_model = RefineModel([18])
  rs_ar_ckpt = torch.load(rs_model_path.format('Ar'), map_location='cpu')
  rs_ar_model.load_state_dict(rs_ar_ckpt['model'])
  rs_ar_model.eval()  
  rs_ar_model.to(device)

  rs_go_model = RefineModel([9])
  rs_go_ckpt = torch.load(rs_model_path.format('Go'), map_location='cpu')
  rs_go_model.load_state_dict(rs_go_ckpt['model'])
  rs_go_model.eval()  
  rs_go_model.to(device)

  # Arguments for looping over test1 and test2
  with torch.no_grad():
    for t_name in test_list:

      gs_rst = gs_dict[t_name]
      ls_rst = ls_dict[t_name]
      rs_rst = rs_dict[t_name]

      for b_idx in range(ls_rst['x'].shape[0]):
        inputs = torch.cat([
          gs_rst['x'][b_idx], ls_rst['x'][b_idx],
          gs_rst['y'][b_idx], ls_rst['y'][b_idx],
          ], dim=0).to(device)
        
        # Predict x^P with linear filters
        ar_rst = rs_ar_model(inputs)
        go_rst = rs_go_model(inputs)

        # Save results
        rs_rst['x'][b_idx:, Ar_IDX] = ar_rst[0]
        rs_rst['y'][b_idx:, Ar_IDX] = ar_rst[1]
        rs_rst['x'][b_idx:, Go_IDX] = go_rst[0]
        rs_rst['y'][b_idx:, Go_IDX] = go_rst[1]

  return rs_dict

   
def save_test_result(args, stage, output_dir, test_list, stat_dict, rst_dict):
  """ Save test results in txt and mat files.
  Args:
    args (argparse): Arguments for experiments
    stage (str): Stage names ('global' / 'local')
    output_dir (str): Output dir 
    test_list (list): List of test names (str)
    stat_dict (dictionary): Test1, Test2 statistics
      Shape = {'test1':{'MRE':mre}, {'SD':sd}, {'SDR':sdr}, {'SCR':scr},
               'test2':{'MRE':mre}, {'SD':sd}, {'SDR':sdr}, {'SCR':scr}}
        mre (dictionary): Test1, Test2 MRE
          Shape = {'A':tensor, ... 'average':tensor}
        sd (dictionary): Test1, Test2 SD
          Shape = {'A':tensor, ... 'average':tensor}     
        sdr (dictionary): Test1, Test2 SDR
          Shape = {20:{'A':tensor, ...}, ... 40:{'A':tensor, ...}
        scr (tensor): Test1, Test2 SCR
          Shape = tensor([1, 8])
    rst_dict (dicionary): Results from each stage
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}    
  """
  print_stat(test_list=test_list, stat_dict=stat_dict)
  write_result(
    exp_str=stage, 
    test_list=test_list, 
    args=args, 
    output_dir=output_dir, 
    stat_dict=stat_dict
  )
  for t_name in test_list:
    save_as_mat(
      mat_data=rst_dict[t_name], 
      mat_name=os.path.join(output_dir, '{}_{}'.format(stage, t_name))
    )


def parse_args():
  """ Arguments for script """
  parser = argparse.ArgumentParser(description='MPA Test')

  # Environmental settings
  parser.add_argument('--gpu_num', default='0', type=str)
  parser.add_argument('--device', default='cuda', help='device')
  parser.add_argument('--workers', default=16, type=int)
  parser.add_argument('--output_dir', default='./output')

  # Test settings
  parser.add_argument('--data_root', default='./data', type=str)

  # Global stages settings
  parser.add_argument('--gs_model_name', default='deeplabv3_resnet50')
  parser.add_argument('--gs_model_path', default='./models/global/All.pth')
  parser.add_argument('--gs_aux_loss', action='store_false')
  parser.add_argument('--gs_img_h', default=720, type=int)
  parser.add_argument('--gs_img_w', default=580, type=int)

  # Local stages settings
  parser.add_argument('--ls_model_name', default='deeplabv3_resnet50')
  parser.add_argument('--ls_model_path', default='./models/local/{}.pth')
  parser.add_argument('--ls_aux_loss', action='store_false')
  parser.add_argument('--pat_sz', default=256, type=int)

  # Refinement stages settings
  parser.add_argument('--rs_model_path', default='./models/refine/{}.pth')
  args = parser.parse_args()

  return args


def main(args):

  # Environmental settings
  print(args)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
  torch.multiprocessing.set_sharing_strategy('file_system')
  device = torch.device(args.device) 
  time_stamp = datetime.today().strftime('%y%m%d_%H%M%S')
  output_dir = os.path.join(args.output_dir, 'test_{}'.format(time_stamp))
  test_list = ['test1', 'test2']
  batch_sz_list = [1, 1]

  print('+---- Global Stage -----')
  gs_dict, gt_dict = global_stage(
    gs_model_name=args.gs_model_name, 
    gs_model_path=args.gs_model_path,
    gs_aux_loss=args.gs_aux_loss, 
    gs_img_h=args.gs_img_h, 
    gs_img_w=args.gs_img_w, 
    device=device, 
    test_list=test_list, 
    loader_dict=get_loader_dict(
      stage='global', 
      gs_img_h=args.gs_img_h,
      gs_img_w=args.gs_img_w,
      data_root=args.data_root,
      num_workers=args.workers,
      test_list=test_list,
      batch_sz_list=batch_sz_list,
    ),
  )
  gs_stat = statistics(test_list=test_list, rst_dict=gs_dict, gt_dict=gt_dict)
  save_test_result(args, 'global', output_dir, test_list, gs_stat, gs_dict)
  print('+-----------------------')

  print('+---- Local Stage -----')
  ls_dict = local_stage(
    ls_model_name=args.ls_model_name, 
    ls_model_path=args.ls_model_path,
    ls_aux_loss=args.ls_aux_loss, 
    pat_sz=args.pat_sz,
    gs_dict=gs_dict, 
    test_list=test_list, 
    device=device,
    loader_dict=get_loader_dict(
      stage='local', 
      gs_img_h=args.gs_img_h,
      gs_img_w=args.gs_img_w,
      data_root=args.data_root,
      num_workers=args.workers,
      test_list=test_list,
      batch_sz_list=batch_sz_list,
    ),
  ) 
  ls_stat = statistics(test_list=test_list, rst_dict=ls_dict, gt_dict=gt_dict)
  save_test_result(args, 'local', output_dir, test_list, ls_stat, ls_dict)
  print('+----------------------')

  print('+---- Refinement Stage -----')
  rs_dict = refine_stage(
    rs_model_path=args.rs_model_path,
    test_list=test_list,
    gs_dict=gs_dict, 
    ls_dict=ls_dict, 
    device=device
  )
  rs_stat = statistics(test_list=test_list, rst_dict=rs_dict, gt_dict=gt_dict)
  save_test_result(args, 'refine', output_dir, test_list, rs_stat, rs_dict)
  print('+----------------------')


if __name__ == "__main__":
  args = parse_args()
  main(args)