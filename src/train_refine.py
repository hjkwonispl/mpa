# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#

# Import global packages
import os
import copy
import argparse
import scipy.io as sio
import torch
from datetime import datetime

# Import global constants
from constants import *

# Import user defined functions
import evaluation as evl
from models import RefineModel
from utils import save_as_mat, print_stat, get_loader_dict
from test import global_stage, local_stage, get_loader_dict


def save_mat_result(stage, output_dir, test_list, stat_dict, rst_dict):
  """ Save result from global and local stages as MATLAB mat files
  Args: 
    output_dir (str): mat file directories
    stage (str): 'global' / 'local' stage
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
  if stat_dict != None:
    print_stat(test_list=test_list, stat_dict=stat_dict)
  for t_name in test_list:
    save_as_mat(
      mat_data=rst_dict[t_name], 
      mat_name=os.path.join(output_dir, '{}_{}'.format(stage, t_name))
    )


def get_global_local_stage_data(
  gs_model_name,
  gs_model_path,
  gs_aux_loss,
  gs_img_h,
  gs_img_w,
  ls_model_name,
  ls_model_path,
  ls_aux_loss,
  pat_sz,
  data_root,
  mat_dir,
  num_workers,
  device,
  ):
  """ Save ground truth, global and local stage results as mat 
  
  Args:  
    gs_model_name (str): Name of torchvision.models for global stage to be used
    gs_model_path (str): Path of trained model for global stage (.pth file)
    gs_aux_loss (bool): Use of aux_loss for global stage (default is True)
    gs_img_h (int): Size of downsampled image (height)
    gs_img_w (int): Size of downsampled image (width)
    ls_model_name (str): Name of torchvision.models for local stage to be used
    ls_model_path (str): Path of trained model for local stage (.pth file)
    ls_aux_loss (bool): Use of aux_loss for local stage (default is True)
    pat_sz (int): Side length of patch image
    data_root (str): Directory of ISBI2015 dataset
    mat_dir (str): mat file directories
    workers (int): Number of theads for dataloader
    device (str): Cuda or CPU   
  """

  # Environmental settings
  os.makedirs(mat_dir, exist_ok=True)
  test_list = ['train', 'test1', 'test2']
  batch_sz_list = [1, 1, 1]

  print('+---- Global Stage -----')
  gs_dict, gt_dict = global_stage(
    gs_model_name=gs_model_name, 
    gs_model_path=gs_model_path,
    gs_aux_loss=gs_aux_loss, 
    gs_img_h=gs_img_h, 
    gs_img_w=gs_img_w,     
    device=device, 
    test_list=test_list, 
    loader_dict=get_loader_dict(
      stage='global', 
      gs_img_h=gs_img_h, 
      gs_img_w=gs_img_w, 
      data_root=data_root,
      num_workers=num_workers,
      test_list=test_list,
      batch_sz_list=batch_sz_list,
    ),
  )
  gs_stat = evl.statistics(test_list, gs_dict, gt_dict)
  save_mat_result('global', mat_dir, test_list, gs_stat, gs_dict)
  save_mat_result('gt', mat_dir, test_list, None, gt_dict)
  print('+-----------------------')

  print('+---- Local Stage -----')
  ls_dict = local_stage(
    ls_model_name=ls_model_name, 
    ls_model_path=ls_model_path,
    ls_aux_loss=ls_aux_loss, 
    pat_sz=pat_sz,
    gs_dict=gs_dict, 
    test_list=test_list, 
    device=device,
    loader_dict=get_loader_dict(
      stage='local', 
      gs_img_h=gs_img_h,
      gs_img_w=gs_img_w,
      data_root=data_root,
      num_workers=num_workers,
      test_list=test_list,
      batch_sz_list=batch_sz_list,
    ),
  ) 

  ls_stat = evl.statistics(test_list, ls_dict, gt_dict)
  save_mat_result('local', mat_dir, test_list, ls_stat, ls_dict)
  print('+----------------------')


def validation(blm_idx, rst, gt):
  """ Validation of one epoch
  
  Args:
    blm_idx (int): Index for targeted bilateral landmark of the mandible
    rst (torch.tensor): Predicted landmark positions. 
      Shape = [n_batch, 2]
    gt (dictionary): Ground truth labels.
      Shape = {'x': tensor([n_batch, NUM_LM]), 'y': tensor([n_batch, NUM_LM])}

  Returns:
    dictionary of results
      Shape: {'MRE': {'A':tensor, ... 'average':tensor}
              'SDR': {20:{'A':tensor, ...}, ... 40:{'A':tensor, ...}
              'SD': {'A':tensor, ... 'average':tensor}}
  """
  rad_err = torch.tensor([]) # RadialError for each test
  gt_x_list = gt['x'][:, [blm_idx]]
  gt_y_list = gt['y'][:, [blm_idx]]
  rst_x_list = rst[:, [0]]
  rst_y_list = rst[:, [1]]

  # Prediction
  for b_idx in range(gt['x'].shape[0]):
    gt_x = gt_x_list[b_idx]
    gt_y = gt_y_list[b_idx]
    rst_x = rst_x_list[b_idx]
    rst_y = rst_y_list[b_idx]
    rad_err_per_sample = evl.radial_error(
      img_batch=torch.zeros([1, 1, RAW_IMG_H, RAW_IMG_W]),
      gt_x=gt_x, gt_y=gt_y, exp_x=rst_x, exp_y=rst_y).unsqueeze(0)
    rad_err = torch.cat([rad_err, rad_err_per_sample], dim=0)
         
  # Calculate statistics
  mre = evl.stats_to_dict(rad_err, 'MRE')
  sd = evl.stats_to_dict(rad_err, 'SD')
  sdr = evl.stats_to_dict(evl.detection_with_bdd(rad_err).float(), 'SDR')
  
  return {'MRE':mre, 'SDR':sdr, 'SD':sd}


def train(
  mat_dir,
  output_dir,
  ckpt_dir,
  blm_idx,
  lr,
  weight_decay,
  device,
  epoch,
  ):
  """ Train refinement stage
  
  Args: 
    mat_dir (str): mat file directories
    output_dir (str): Output dir 
    ckpt_dir (str): Checkpoint directory for trained models
    blm_idx (int): Index for targeted bilateral landmark of the mandible
    lr (float): Learning rate
    weight_decay (float): Weight_decay
    device (str): Cuda or CPU   
    epoch (int): Total training epoch
  """
  # Load training data (prediction from global, local stages)
  gs_tr_mat = sio.loadmat(os.path.join(mat_dir, 'global_train'))
  gs_t1_mat = sio.loadmat(os.path.join(mat_dir, 'global_test1'))
  gs_t2_mat = sio.loadmat(os.path.join(mat_dir, 'global_test2'))
  ls_tr_mat = sio.loadmat(os.path.join(mat_dir, 'local_train'))
  ls_t1_mat = sio.loadmat(os.path.join(mat_dir, 'local_test1'))
  ls_t2_mat = sio.loadmat(os.path.join(mat_dir, 'local_test2'))
  gt_tr_mat = sio.loadmat(os.path.join(mat_dir, 'gt_train'))
  gt_t1_mat = sio.loadmat(os.path.join(mat_dir, 'gt_test1'))
  gt_t2_mat = sio.loadmat(os.path.join(mat_dir, 'gt_test2'))

  gs_tr = {'x':torch.tensor(gs_tr_mat['x']), 'y':torch.tensor(gs_tr_mat['y'])}
  gs_t1 = {'x':torch.tensor(gs_t1_mat['x']), 'y':torch.tensor(gs_t1_mat['y'])}
  gs_t2 = {'x':torch.tensor(gs_t2_mat['x']), 'y':torch.tensor(gs_t2_mat['y'])}
  ls_tr = {'x':torch.tensor(ls_tr_mat['x']), 'y':torch.tensor(ls_tr_mat['y'])}
  ls_t1 = {'x':torch.tensor(ls_t1_mat['x']), 'y':torch.tensor(ls_t1_mat['y'])}
  ls_t2 = {'x':torch.tensor(ls_t2_mat['x']), 'y':torch.tensor(ls_t2_mat['y'])}
  gt_tr = {'x':torch.tensor(gt_tr_mat['x']), 'y':torch.tensor(gt_tr_mat['y'])}
  gt_t1 = {'x':torch.tensor(gt_t1_mat['x']), 'y':torch.tensor(gt_t1_mat['y'])}
  gt_t2 = {'x':torch.tensor(gt_t2_mat['x']), 'y':torch.tensor(gt_t2_mat['y'])}

  # Load models
  model = RefineModel([blm_idx])
  model.to(device)
  params_to_optimize = model.get_train_params(lr)

  # Optimizer and leraning rate scheduler.
  optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda x: (1 - x / (SZ_TRAINING * 1e7)) ** weight_decay
  ) 

  # Training
  model.train()
  for step in range(SZ_TRAINING * epoch):
    n_train_data = gs_tr['x'].shape[0]

    for b_idx in range(n_train_data):
      inputs = torch.cat([
        gs_tr['x'][b_idx], ls_tr['x'][b_idx],
        gs_tr['y'][b_idx], ls_tr['y'][b_idx],
        ], dim=0).to(device)
      targets = torch.cat([
        gt_tr['x'][b_idx][[blm_idx]], 
        gt_tr['y'][b_idx][[blm_idx]]], 
        dim=0
      ).float().to(device)
      outputs = model(inputs)

      loss = model.calc_loss(preds=outputs, gt=targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      for p in model.parameters(): # Regulization 
          p.data.clamp_(0) # Make parameters larger than 0
      
      lr_scheduler.step() 

    # Validation
    model.eval()
    n_test_data = gs_t1['x'].shape[0]
    test_loss = 0
    rs_t1_rst = []
    with torch.no_grad():
      for b_idx in range(n_test_data):
        inputs = torch.cat([
          gs_t1['x'][b_idx], ls_t1['x'][b_idx],
          gs_t1['y'][b_idx], ls_t1['y'][b_idx],
          ], dim=0).to(device)
        targets = torch.cat([
          gt_t1['x'][b_idx], gt_t1['y'][b_idx]
          ], dim=0).float().to(device)
        rs_t1_rst.append(model(inputs).detach().clone().cpu().unsqueeze(0))
    rs_t1_rst = torch.cat(rs_t1_rst, dim=0)
    val_dict = validation(blm_idx, rs_t1_rst, gt_t1)

    # Save best model
    if step == 0:
      best_val_stat = {
        'MRE': val_dict['MRE'], 
        'SDR': val_dict['SDR'], 
        'step': step
      }
    else:
      best_val_mre = best_val_stat['MRE']['average']
      cur_val_mre = val_dict['MRE']['average']
      if best_val_mre > cur_val_mre:
        best_val_stat['MRE'] = val_dict['MRE']
        best_val_stat['SDR'] = val_dict['SDR']
        best_val_stat['step'] = step
        torch.save({
          'model': model.state_dict(),},
          os.path.join(ckpt_dir,
            'best_{}.pth'.format(S_LM_NAME_DICT[blm_idx])
          )
        )  

    # Logging    
    print('Landmark: {}, Step: {}, MRE: {:04f}, SDR: {:04f}'.format(
      S_LM_NAME_DICT[blm_idx],
      step,
      val_dict['MRE']['average'], 
      val_dict['SDR'][20]['average']
    ))
    log_rs(
      step=step, 
      output_dir=output_dir, 
      best_val_stat=best_val_stat, 
      val_dict=val_dict,
    )


def log_rs(step, output_dir, best_val_stat, val_dict):
  """ Save test results in txt and mat files for local stage.
  Args:
    step (int): training step
    output_dir (str): Output dir 
    best_val_stat (dictionary): Test1, Test2 best results till this 'step'
      Shape = {'test1':{'MRE':mre}, {'SDR':sdr}, {'step':(int)},
               'test2':{'MRE':mre}, {'SDR':sdr}, {'step':(int)}}
        mre (dictionary): Test1, Test2 MRE
          Shape = {'A':tensor, ... 'average':tensor}
        sdr (dictionary): Test1, Test2 SDR
          Shape = {20:{'A':tensor, ...}, ... 40:{'A':tensor, ...}
    test_dict (dictionary): Test1, Test2 statistics
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
  """
  with open(os.path.join(output_dir, 'summary.txt'), 'a+') as summary_file:
    if step == 0:
      summary_file.write(str(args) + '\n')
      summary_file.write('step\tTest1\tBest_Step\tBest_MRE\tMRE\t')
      summary_file.write('SDR20\tSDR25\tSDR30\tSDR40\n')
    summary_file.write('{}\t'.format(step))

    summary_file.write('{}\t{}\t{:.4f}\t{:.4f}\t'.format(
      'test1', 
      best_val_stat['step'],
      best_val_stat['MRE']['average'],
      val_dict['MRE']['average'],    
    ))
    summary_file.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t'.format(
      best_val_stat['SDR'][20]['average'],
      best_val_stat['SDR'][25]['average'],
      best_val_stat['SDR'][30]['average'],
      best_val_stat['SDR'][40]['average'],
    ))
    summary_file.write('\n')


def main(args):
  # Environmental Settings.
  print(args)
  time_stamp = datetime.today().strftime('%y%m%d_%H%M%S')
  exp_key_str = 'refine_{}'.format(time_stamp)
  output_dir = os.path.join(args.output_dir, exp_key_str)
  ckpt_dir = args.ckpt_dir
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(ckpt_dir, exist_ok=True)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
  torch.multiprocessing.set_sharing_strategy('file_system')
  device = torch.device(args.device)

  # Get global and local stage results into MATLAB mat file
  get_global_local_stage_data(
    gs_model_name=args.gs_model_name,
    gs_model_path=args.gs_model_path,
    gs_aux_loss=args.gs_aux_loss,
    gs_img_h=args.gs_img_h,
    gs_img_w=args.gs_img_w,
    ls_model_name=args.ls_model_name,
    ls_model_path=args.ls_model_path,
    ls_aux_loss=args.ls_aux_loss,
    pat_sz=args.pat_sz,
    data_root=args.data_root,
    mat_dir=args.mat_dir,
    num_workers=args.workers,
    device=device,
  )

  # Train refinement stage model
  train(
    mat_dir=args.mat_dir,
    output_dir=output_dir,
    ckpt_dir=ckpt_dir,
    blm_idx=args.blm_idx,
    lr=args.lr,
    weight_decay=args.weight_decay,
    device=device,
    epoch=args.epoch,
  )


def parse_args():
  """ Arguments for script """
  parser = argparse.ArgumentParser(description='Refinement Stage Training')

  # Environmental settings
  parser.add_argument('--gpu_num', '-gn', default='0', type=str)
  parser.add_argument('--device', default='cuda', help='device')
  parser.add_argument('--workers', '-j', default=16, type=int)
  parser.add_argument('--output_dir', default='./output')
  parser.add_argument('--mat_dir', default='./mat')
  parser.add_argument('--ckpt_dir', default='./ckpt/refine', type=str)
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

  # Training settings
  parser.add_argument('--epoch', default=5, type=int)
  parser.add_argument('--lr', default=0.000001, type=float)
  parser.add_argument('--weight_decay', default=0.9, type=float)
  parser.add_argument('--blm_idx', default=9, type=int)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = parse_args()
  main(args)
