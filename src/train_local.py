# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#

# Import global packages
import os
import argparse
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from kornia.geometry.subpix import spatial_soft_argmax2d

# Import global constants
from constants import *

# Import user defined classes and functions
from isbi_dataset import crop_lm_patches
from evaluation import statistics
from test import global_stage
from isbi_dataset import ISBIDataSet, ann_to_heatmap, vis_patch, augmentation
from evaluation import radial_error, stats_to_dict, detection_with_bdd
from models import TVSegModel
from utils import save_model, get_loader_dict, print_stat

# For tensorboard
writer = SummaryWriter() 


def ls_tb_val_scalar(tb_str, epoch, mre, sdr):
  """ Local stage tensorboard outputs for validation (scalars)
  Args:
    tb_str (str): Additional information for tensorboard
    epoch (int): Global step for summary writer
    mre (dict): Dictionary of MRE for landmarks and average
    sdr (dict): Dictionary of SDR for landmarks and average for each threshold
  """
  writer.add_scalar('00_MRE/' + tb_str, mre['average'], epoch)
  writer.add_scalar('01_SDR20/' + tb_str, sdr[20]['average'], epoch)
  writer.add_scalar('02_SDR25/' + tb_str, sdr[25]['average'], epoch)
  writer.add_scalar('03_SDR30/' + tb_str, sdr[30]['average'], epoch)
  writer.add_scalar('04_SDR40/' + tb_str, sdr[40]['average'], epoch)


def ls_tb_train_scalar(loss, step):
  """ Local stage tensorboard outputs for training (scalars)
  Args:
    loss (str): Training loss
    step (int): Global step for summary writer
  """
  writer.add_scalar('00_Loss/train', loss, step)


def ls_tb_val_image(tb_str, epoch, img, heatmap, exp_x, exp_y, lm_idx):
  """ Local stage tensorboard outputs for validation (image)
  Args:
    tb_str (str): Additional information for tensorboard
    epoch (int): Global step for summary writer
    img (tensor): Input image
    heatmap (tensor): P^L
    exp_x (tensor): Predicted 'x' coordinates for landmarks
    exp_y (tensor): Predicted 'y' coordinates for landmarks
  """
  lm_img = vis_patch(img_batch=img, 
      x=exp_x, y=exp_y, c=torch.zeros([8]),
      radius=5, font_scale=.8, txt_offset=15, lm_idx=lm_idx)[0, :]

  output_img = torch.clamp(heatmap, min=0, max=1).transpose(0, 1)[0, :]
  writer.add_image('01_LM/' + tb_str, lm_img, epoch)
  writer.add_image('02_HM/' + tb_str, output_img, epoch)


def ls_tb_train_image(img_batch, gt_x, gt_y, lm_idx, output, epoch):
  """ Global stage tensorboard outputs for training (images)
  Args:
    img_batch (str): Input image (Patch)
    gt_x (int): Ground truth annotation for the patch (x coordinate)
    gt_x (int): Ground truth annotation for the patch (y coordinate)
    lm_idx (int): Index of target landmark 
    output (tensor): P^L
    epoch (int): Global step for summary writer
  """  
  # Log output
  writer.add_image('Train/01_GT',
    img_tensor=vis_patch(
        img_batch=img_batch, 
        x=gt_x, y=gt_y, c=None, 
        radius=5, font_scale=0.8, txt_offset=15,
        lm_idx=lm_idx,
    )[0, :],
    global_step=epoch,
  )
  writer.add_image('Train/02_P^L',
    img_tensor=torch.clamp(output, 0, 1)[0, :],
    global_step=epoch,
  )


# For point regression
def validation(
  model, 
  pat_sz, 
  test_list,
  loader_dict, 
  gs_dict, 
  lm_idx,
  device, 
  epoch
  ):
  """ Validation with P^L and x^L
  
  Args:  
    model (torch.nn.Module): Network of global stage
    pat_sz (int): Side length of crooped patch image
    test_list (list): List of test names (str)
    loader_dict (dict): Dictionary of data_loader (key:test_name)
    lm_idx (int): Index of target landmark 
    device (str): Cuda or CPU  
    epoch (int): training epoch
    gs_dict (dictionary): Test1, Test2 results from global stage
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}
  
  Returns:
    test_dict (dictionary): Test1, Test2 statistics of global stage
      Shape = {'test1':{'MRE':mre}, {'SDR':sdr},
               'test2':{'MRE':mre}, {'SDR':sdr}}
  """
  model.eval()
  test_dict = {}

  for t_name in test_list:
    
    t_loader = loader_dict[t_name]
    rad_err = torch.tensor([])
    
    with torch.no_grad():
      for b_idx, (img_batch, ann_batch) in enumerate(t_loader):
        img_batch = img_batch.to(device)

        # Use global stage output for crop center
        gs_x = gs_dict[t_name]['x'][b_idx][lm_idx:lm_idx+1]
        gs_y = gs_dict[t_name]['y'][b_idx][lm_idx:lm_idx+1]
        ann_batch = {k:v[:, lm_idx] for k, v in ann_batch.items()}
        
        # Crop patch and calculate ground truth landmark position in the patch
        pat_img_batch, pat_ana_x, pat_ana_y = crop_lm_patches(
          img_batch=img_batch, 
          ann_batch=ann_batch, 
          pat_sz=pat_sz, 
          x_c_batch=gs_x,
          y_c_batch=gs_y,
        )
        test_img_batch = pat_img_batch.to(device)
        gt_x = pat_ana_x.float()
        gt_y = pat_ana_y.float()

        # Predict P^L
        output = model(test_img_batch)
        
        # Calcuate x^L
        exp_xy = spatial_soft_argmax2d(
          output['out'], 
          normalized_coordinates=False
        )
        exp_x = exp_xy[:, :, 0].cpu()
        exp_y = exp_xy[:, :, 1].cpu()

        # Calculate radial error
        rad_err_per_sample = radial_error(img_batch=img_batch,
          gt_x=gt_x, gt_y=gt_y, exp_x=exp_x, exp_y=exp_y)
        rad_err = torch.cat([rad_err, rad_err_per_sample], dim=0)
        
    # Average statistics with names.
    mre = stats_to_dict(rad_err, 'MRE')
    sdr = stats_to_dict(detection_with_bdd(rad_err).float(), 'SDR')
    
    # Store into dictionary
    test_dict[t_name] = {'MRE': mre, 'SDR': sdr}

    # To tensorboard
    ls_tb_val_scalar(tb_str=t_name, epoch=epoch, mre=mre, sdr=sdr)
    ls_tb_val_image(
      tb_str=t_name, 
      epoch=epoch, 
      img=test_img_batch, 
      heatmap=output['out'], 
      exp_x=exp_x, 
      exp_y=exp_y, 
      lm_idx=lm_idx,
    )

  return test_dict


def train_one_epoch(
  model,
  optimizer, 
  data_loader, 
  aug_freq,
  lr_scheduler, 
  device, 
  lm_idx,
  rand_dist,
  pat_sz,
  epoch):
  """ Training each epoch for P^L
  
  Args:  
    model (torch.nn.Module): Network of global stage
    optimizer (torch.optim): List of test names (str)
    data_loader (torch.utils.data.dataloader): Dataloader for training
    aug_freq (float): Data augmentation frequency [0.0, 1.0]
    lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
    lm_idx (int): Index of target landmark 
    rand_dist (int): The maximum absolute value of random perturbation
    pat_sz (int): Side length of patch image
    device (str): Cuda or CPU  
    epoch (int): training epoch
  """
  model.train()

  for b_idx, (raw_img_batch, ann_batch) in enumerate(data_loader):
    
    # LM point on raw image
    ann_batch = {
        k:v[:, lm_idx:lm_idx + 1] for k, v in ann_batch.items()
    }
    
    # Perturb lm points
    x_perturb = torch.randint_like(ann_batch['ana_x'], -rand_dist, rand_dist)
    y_perturb = torch.randint_like(ann_batch['ana_y'], -rand_dist, rand_dist)
    x_c_batch = torch.clamp(ann_batch['ana_x'] - x_perturb, 0, RAW_IMG_W)
    y_c_batch = torch.clamp(ann_batch['ana_y'] - y_perturb, 0, RAW_IMG_H)
    
    # Generate training patches and GTs
    pat_img_batch, pat_ana_x, pat_ana_y = crop_lm_patches(
      img_batch=raw_img_batch.float(), 
      ann_batch=ann_batch, 
      pat_sz=pat_sz, 
      x_c_batch=x_c_batch.float(),
      y_c_batch=y_c_batch.float(),
    )

    # Generate heatmap for augmentation
    hm_batch = ann_to_heatmap(pat_img_batch,
      ksize=1,
      sigma=1,
      x=pat_ana_x.float(),
      y=pat_ana_y.float(),
      c=None,
    ).float()

    # Data Augmentation
    if (torch.randint(0, 100, (1,)) < int(aug_freq * 100)):
      pat_img_batch, _, pat_ana_x, pat_ana_y, _, _  = augmentation(
        img_batch=pat_img_batch,
        heatmap_batch=hm_batch,
        degrees=[-25, 25],
        x=pat_ana_x.float(),
        y=pat_ana_y.float(),
        scale=[0.9, 1.2],
        brightness=0.25,
        contrst=0.25,
        saturation=0.25,
        hue=0.25,
        same_on_batch=False,
      )

    # Place to device
    train_img_batch = pat_img_batch.to(device)
    gt_x = pat_ana_x.float().to(device)
    gt_y = pat_ana_y.float().to(device)

    # Pred P^L and backprop with x^L
    output = model(train_img_batch)
    loss = model.calc_pat_loss(
      preds=output, 
      gt_x=gt_x,
      gt_y=gt_y,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    # Log output
    ls_tb_train_scalar(loss=loss, step=b_idx + epoch * len(data_loader))

  # Log output
  ls_tb_train_image(train_img_batch, gt_x, gt_y, lm_idx, output['out'], epoch)


def gs_offset(
  gs_model_name,
  gs_model_path,
  gs_aux_loss, 
  gs_img_h, 
  gs_img_w, 
  data_root,
  workers,
  device):
  """Global stage (calculate P^G and x^G).
  
  Args:  
    gs_model_name (str): Name of torchvision.models for global stage to be used
    gs_model_path (str): Path of trained model for global stage (.pth file)
    gs_aux_loss (bool): Use of aux_loss for global stage (default is True)
    gs_img_h (int): Size of downsampled image (height)
    gs_img_w (int): Size of downsampled image (width)
    data_root (str): Directory of ISBI2015 dataset
    workers (int): Number of theads for dataloader
    device (str): Cuda or CPU  
  
  Returns:
    gs_dict (dictionary): Test1, Test2 results from global stage
      Shape = {'test1':{'x':[...]}, {'y':[...]},
               'test2':{'x':[...]}, {'y':[...]},
               ...}
  """
  print('+---- Global Stage -----')
  test_list = ['test1', 'test2']
  batch_sz_list = [1, 1]
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
      num_workers=workers,
      test_list=test_list,
      batch_sz_list=batch_sz_list,
    ),
  )
  gs_stat = statistics(test_list, gs_dict, gt_dict)
  print_stat(test_list, gs_stat)
  print('+-----------------------')

  return gs_dict


def main(args):
  # Environmental Settings.
  print(args)
  global writer # Tensorboard writer
  time_stamp = datetime.today().strftime('%y%m%d_%H%M%S')
  exp_key_str = '{}_{}_{}_R{}_P{}'.format(
    'local', time_stamp, S_LM_NAME_DICT[args.lm_idx],
    args.rand_dist, args.pat_sz
  )
  output_dir = os.path.join(args.output_dir, exp_key_str)
  ckpt_dir = args.ckpt_dir
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(ckpt_dir, exist_ok=True)
  writer = SummaryWriter(comment='_{}_R{}_P{}'.format(
    S_LM_NAME_DICT[args.lm_idx],args.rand_dist, args.pat_sz
  ))

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
  device = torch.device(args.device)

  # Get DataLoader
  loader_dict = get_loader_dict(
      stage='local', 
      data_root=args.data_root,
      num_workers=args.workers,
      gs_img_h=args.gs_img_h, 
      gs_img_w=args.gs_img_w, 
      test_list=['train', 'test1', 'test2'],
      batch_sz_list=[args.batch_size, 1, 1],
  )

  # Get stage1 test data
  gs_dict = gs_offset(
    gs_model_name=args.gs_model_name,
    gs_model_path=args.gs_model_path,
    gs_aux_loss=args.gs_aux_loss, 
    gs_img_h=args.gs_img_h, 
    gs_img_w=args.gs_img_w, 
    data_root=args.data_root,
    workers=args.workers,
    device=device
  )

  # Model and trainable parameters.
  model = TVSegModel(
    tv_model=args.model_name, 
    aux_loss=args.aux_loss, 
    in_ch=3, 
    num_classes=1
  )
  model.to(device)
  params_to_optimize = model.get_train_params(args.lr)

  # Optimizer and leraning rate scheduler.
  optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, 
    weight_decay=args.weight_decay
  )
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda x: (1 - x / (len(loader_dict['train']) * args.epochs)) ** 0.9
  )

  # Resume training from checkpoint
  if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    args.start_epoch = checkpoint['epoch']
  
  # Train and evaluation
  best_val_stat = {}
  for epoch in range(args.start_epoch, args.epochs):

    train_one_epoch(
      epoch=epoch, 
      model=model, 
      device=device, 
      data_loader=loader_dict['train'],
      optimizer=optimizer, 
      lr_scheduler=lr_scheduler,
      aug_freq=args.aug_freq, 
      pat_sz=args.pat_sz, 
      lm_idx=args.lm_idx, 
      rand_dist=args.rand_dist,
    )

    test_dict = validation(
      epoch=epoch, 
      model=model, 
      device=device, 
      test_list=['test1', 'test2'],
      loader_dict=loader_dict,
      pat_sz=args.pat_sz, 
      lm_idx=args.lm_idx,
      gs_dict=gs_dict,
    )

    # Save best_test_model
    if epoch == 0:
      best_val_stat = {
        'MRE': test_dict['test1']['MRE'], 
        'SDR': test_dict['test1']['SDR'], 
        'epoch': epoch
      }
    else:
      best_val_mre = best_val_stat['MRE']['average']
      cur_val_mre = test_dict['test1']['MRE']['average']
      if best_val_mre > cur_val_mre:
        best_val_stat['MRE'] = test_dict['test1']['MRE']
        best_val_stat['SDR'] = test_dict['test1']['SDR']
        best_val_stat['epoch'] = epoch
        save_model(model, optimizer, lr_scheduler, epoch, ckpt_dir,
          'best_{}.pth'.format(S_LM_NAME_DICT[args.lm_idx])
        )
      print('Epoch={}, Dataset={}, Aug_Freq={}, MRE={:.4f}'.format(
        epoch, args.data_root, args.aug_freq, best_val_mre))

    # Save training model
    if epoch % args.save_freq == 0:
      save_model(model, optimizer, lr_scheduler, epoch, ckpt_dir,
        '{}_{}.pth'.format(S_LM_NAME_DICT[args.lm_idx], epoch)
      )

    # Logging
    log_ls(
      epoch=epoch, 
      output_dir=output_dir,
      best_val_stat=best_val_stat, 
      test_list=['test1', 'test2'],
      test_dict=test_dict
    )


def log_ls(epoch, output_dir, best_val_stat, test_list, test_dict):
  """ Save test results in txt and mat files for local stage.
  Args:
    epoch (int): training epoch
    output_dir (str): Output dir 
    test_list (list): List of test names (str)
    best_val_stat (dictionary): Test1, Test2 best results till this 'epoch'
      Shape = {'test1':{'MRE':mre}, {'SDR':sdr}, {'epoch':(int)},
               'test2':{'MRE':mre}, {'SDR':sdr}, {'epoch':(int)}}
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
  def write_ls_stat_dict(epoch, stat_file_name, stat_dict):
    with open(stat_file_name, 'a+') as stat_file:
      if epoch == 0:
        stat_file.write('Epoch\t')
        for key, _ in stat_dict.items():
          stat_file.write('{}\t'.format(key))
        stat_file.write('\n')
      stat_file.write('{}\t'.format(epoch))
      for _, value in stat_dict.items():
        stat_file.write('{:.4f}\t'.format(value))
      stat_file.write('\n')
  
  def write_ls_mre_sdr(eval_str, mre, sdr):
      mre_file_name = os.path.join(output_dir, '{}_mre.txt'.format(eval_str))
      write_ls_stat_dict(epoch, mre_file_name, mre)

      for det_bdd in DET_BDD_LIST:
        sdr_file_name = os.path.join(output_dir,
          '{}_sdr{}.txt'.format(eval_str, det_bdd)
        )
        write_ls_stat_dict(epoch, sdr_file_name, sdr[det_bdd])

  # Write stat for tests
  for t_name in test_list:
    mre = test_dict[t_name]['MRE']
    sdr = test_dict[t_name]['SDR']
    write_ls_mre_sdr(t_name, mre, sdr)

  # Write summary info
  with open(os.path.join(output_dir, 'summary.txt'), 'a+') as summary_file:
    if epoch == 0:
      summary_file.write(str(args) + '\n')
      summary_file.write('Epoch\tTest1\tBest_Epoch\tBest_MRE')
      summary_file.write('\tMRE\tSDR20\tSDR25\tSDR30\tSDR40\t')
      summary_file.write('Test2\tMRE\tSDR20\tSDR25\tSDR30\tSDR40\n')
    summary_file.write('{}\t'.format(epoch))
    
    # For test
    for t_idx, t_name in enumerate(test_list):
      mre = test_dict[t_name]['MRE']
      sdr = test_dict[t_name]['SDR']
      if t_idx == 0:
        summary_file.write('{}\t{}\t{:.4f}\t{:.4f}\t'.format(
          t_name, 
          best_val_stat['epoch'],
          best_val_stat['MRE']['average'],
          mre['average'])
        )
      else:
        summary_file.write('{}\t{:.4f}\t'.format(
          t_name, 
          mre['average'])
        )
      for det_bdd in DET_BDD_LIST:
        summary_file.write('{:.4f}\t'.format(sdr[det_bdd]['average']))
    summary_file.write('\n')


def parse_args():
  parser = argparse.ArgumentParser(description='Peacock Training')

  # Environmental settings
  parser.add_argument('--gpu_num', default='0', type=str)
  parser.add_argument('--device', default='cuda', help='device')
  parser.add_argument('--workers', default=16, type=int)
  parser.add_argument('--ckpt_dir', default='./ckpt/local')
  parser.add_argument('--output_dir', default='./output')
  parser.add_argument('--data_root', default='./data', type=str)
  parser.add_argument('--save_freq', default=30, type=int)

  # Patch settings

  # Local stage settings
  parser.add_argument('--model_name', default='deeplabv3_resnet50')
  parser.add_argument('--aux_loss', action='store_false')
  parser.add_argument('--lm_idx', default=18, type=int)  
  parser.add_argument('--pat_sz', default=256, type=int)
  parser.add_argument('--rand_dist', default=128, type=int)

  # Global stage settings
  parser.add_argument('--gs_model_name', default='deeplabv3_resnet50')
  parser.add_argument('--gs_model_path', default='./models/global/All.pth')
  parser.add_argument('--gs_aux_loss', action='store_false')
  parser.add_argument('--gs_img_h', default=720, type=int)
  parser.add_argument('--gs_img_w', default=580, type=int)

  # Training settings
  parser.add_argument('--aug_freq', default=0.1, type=float) 
  parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
  parser.add_argument('--batch_size', default=2, type=int)
  parser.add_argument('--epochs', default=50, type=int)
  parser.add_argument('--lr', default=0.0001, type=float)
  parser.add_argument('--weight_decay', default=0, type=float)

  # Resume training
  parser.add_argument('--resume', default='', action='store_false')

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main(args)
