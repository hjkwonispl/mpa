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
from isbi_dataset import ISBIDataSet, ann_to_heatmap, vis_isbi, augmentation
from evaluation import radial_error, stats_to_dict, detection_with_bdd
from models import TVSegModel
from utils import save_model, get_loader_dict

# For tensorboard
writer = SummaryWriter() 
  

def gs_tb_val_scalar(tb_str, epoch, mre, sdr):
  """ Global stage tensorboard outputs for validation (scalars)
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


def gs_tb_train_scalar(loss, step):
  """ Global stage tensorboard outputs for training (scalars)
  Args:
    loss (str): Training loss
    step (int): Global step for summary writer
  """
  writer.add_scalar('00_Loss/train', loss, step)


def gs_tb_val_image(tb_str, epoch, img, output, exp_x, exp_y):
  """ Global stage tensorboard outputs for validation (image)
  Args:
    tb_str (str): Additional information for tensorboard
    epoch (int): Global step for summary writer
    img (tensor): Input image
    output (tensor): P^G
    exp_x (tensor): Predicted 'x' coordinates for landmarks
    exp_y (tensor): Predicted 'y' coordinates for landmarks
  """
  sum_img = torch.clamp(output.sum(1), min=0, max=1)
  lm_img = vis_isbi(img_batch=img, pred_batch=output, 
      x=exp_x, y=exp_y, c=torch.zeros([8]),
      radius=5, font_scale=.8, txt_offset=15
  ).squeeze()
  output_img = make_grid(
      tensor=torch.clamp(output, min=0, max=1).transpose(0, 1),
      nrow=5, pad_value=1, padding=10,
  )
  writer.add_image('01_LM/' + tb_str, lm_img, epoch)
  writer.add_image('02_Sum of P^G/' + tb_str, sum_img, epoch)
  writer.add_image('03_P^G/' + tb_str, output_img, epoch)


def gs_tb_train_image(img_batch, ann_batch, output, epoch):
  """ Global stage tensorboard outputs for training (images)
  Args:
    img_batch (str): (Augmented) input image
    ann_batch (int): (Augmented) ground truth annotatiom
    output (tensor): P^G
    epoch (int): Global step for summary writer
  """  
  writer.add_image('Train/01_GT_LM',
    img_tensor=vis_isbi(img_batch=img_batch, pred_batch=img_batch,
      x=ann_batch['ana_x_fs'], y=ann_batch['ana_y_fs'], c=ann_batch['ana_c'],
      radius=5, font_scale=0.8, txt_offset=15
    )[0, :, :, :],
    global_step=epoch,
  )
  writer.add_image('Train/02_P^G',
    img_tensor=make_grid(
      tensor=torch.clamp(output['out'][0, :, :, :], 0, 1).unsqueeze(1),
      nrow=5,
      pad_value=1,
      padding=10,
    ),
    global_step=epoch,
  )


def validation(model, test_list, loader_dict, device, epoch):
  """ Validation with P^G and x^G
  
  Args:  
    model (torch.nn.Module): Network of global stage
    test_list (list): List of test names (str)
    loader_dict (dict): Dictionary of data_loader (key:test_name)
    device (str): Cuda or CPU  
    epoch (int): training epoch
  
  Returns:
    test_dict (dictionary): Test1, Test2 statistics of global stage
      Shape = {'test1':{'MRE':mre}, {'SDR':sdr},
               'test2':{'MRE':mre}, {'SDR':sdr}}
  """
  model.eval()
  test_dict = {}

  for t_name in test_list:

    t_loader = loader_dict[t_name]
    rad_err = torch.tensor([]) # RadialError for each test
    
    with torch.no_grad():
      for b_idx, (img_batch, ann_batch) in enumerate(t_loader):
        img_batch = img_batch.float().to(device)
        output = model(img_batch)
        gt_x = ann_batch['ana_x']
        gt_y = ann_batch['ana_y']

        # Calcuate Radial Error
        exp_xy = spatial_soft_argmax2d(
          output['out'], 
          normalized_coordinates=False
        )
        exp_x = exp_xy[:, :, 0].cpu()
        exp_y = exp_xy[:, :, 1].cpu()

        rad_err_per_sample = radial_error(img_batch=img_batch,
          gt_x=gt_x, gt_y=gt_y, exp_x=exp_x, exp_y=exp_y)
        rad_err = torch.cat([rad_err, rad_err_per_sample], dim=0)
        
    # Average statistics with names.
    mre = stats_to_dict(rad_err, 'MRE')
    sdr = stats_to_dict(detection_with_bdd(rad_err).float(), 'SDR')
    
    # Store into dictionary
    test_dict[t_name] = {'MRE':mre, 'SDR':sdr}

    # Tensorboard Outputs (Scalars)
    gs_tb_val_scalar(tb_str=t_name, epoch=epoch, mre=mre, sdr=sdr)

    # Tensorboard Outputs (Images)
    gs_tb_val_image(
      tb_str=t_name, 
      epoch=epoch, 
      img=img_batch, 
      output=output['out'], 
      exp_x=exp_x,   
      exp_y=exp_y
    )
    
  return test_dict


def train_one_epoch(
  model,
  optimizer, 
  data_loader, 
  aug_freq,
  lr_scheduler, 
  device, 
  epoch):
  """ Training each epoch for P^G
  
  Args:  
    model (torch.nn.Module): Network of global stage
    optimizer (torch.optim): List of test names (str)
    data_loader (torch.utils.data.dataloader): Dataloader for training
    aug_freq (float): Data augmentation frequency [0.0, 1.0]
    lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
    device (str): Cuda or CPU  
    epoch (int): training epoch
  """
  model.train()

  for b_idx, (img_batch, ann_batch) in enumerate(data_loader):
    img_batch = img_batch.float()
    tgt_batch = ann_to_heatmap(img_batch,
      ksize=10,
      sigma=5,
      x=ann_batch['ana_x_fs'],
      y=ann_batch['ana_y_fs'],
      c=ann_batch['ana_c'],
    ).float()

    # Data Augmentation and get ground truth landmark positions of augmented 
    # images
    if (torch.randint(0, 100, (1,)) < int(aug_freq * 100)):
      print('Epoch={}, Batch={}: Augmented'.format(epoch, b_idx))
      img_batch, _, aug_x_fs, aug_y_fs, aug_x, aug_y = augmentation(
        img_batch=img_batch,
        heatmap_batch=tgt_batch,
        degrees=[-25, 25],
        x=ann_batch['ans_x_fs'],
        y=ann_batch['ans_y_fs'],
        scale=[0.9, 1.2],
        brightness=0.25,
        contrst=0.25,
        saturation=0.25,
        hue=0.25,
        same_on_batch=False,
      )
      ann_batch['ana_x'] = aug_x
      ann_batch['ana_y'] = aug_y
      ann_batch['ana_x_fs'] = aug_x_fs
      ann_batch['ana_y_fs'] = aug_y_fs

    img_batch = img_batch.to(device)

    # Predict P^G and compute loss with x^G
    output = model(img_batch)
    loss = model.calc_loss(
      img_batch=img_batch,
      preds=output, 
      gt_x=ann_batch['ana_x'].float().to(device),
      gt_y=ann_batch['ana_y'].float().to(device),
    )

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    # Logging to tensorboard (scalar)
    gs_tb_train_scalar(loss=loss, step=b_idx + epoch * len(data_loader))

  # Logging to tensorboard (image)
  gs_tb_train_image(img_batch, ann_batch, output, epoch)


def log_gs(epoch, output_dir, best_val_stat, test_list, test_dict):
  """ Save test results in txt and mat files for global stage.
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
  def write_gs_stat_dict(epoch, stat_file_name, stat_dict):
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
  
  def write_gs_mre_sdr(eval_str, mre, sdr):
      mre_file_name = os.path.join(output_dir, '{}_mre.txt'.format(eval_str))
      write_gs_stat_dict(epoch, mre_file_name, mre)

      for det_bdd in DET_BDD_LIST:
        sdr_file_name = os.path.join(output_dir,
          '{}_sdr{}.txt'.format(eval_str, det_bdd)
        )
        write_gs_stat_dict(epoch, sdr_file_name, sdr[det_bdd])

  # Write stat for tests
  for t_name in test_list:
    mre = test_dict[t_name]['MRE']
    sdr = test_dict[t_name]['SDR']
    write_gs_mre_sdr(t_name, mre, sdr)

  # Write summary info
  with open(os.path.join(output_dir, 'summary.txt'), 'a+') as summary_file:
    if epoch == 0:
      summary_file.write(str(args) + '\n')
      summary_file.write('epoch\ttest1\tBest_Epoch\tBest_MRE')
      summary_file.write('\tMRE\tSDR20\tSDR25\tSDR30\tSDR40\t')
      summary_file.write('test2\tMRE\tSDR20\tSDR25\tSDR30\tSDR40\n')
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


def main(args):

  # Environmental Settings.
  print(args)
  time_stamp = datetime.today().strftime('%y%m%d_%H%M%S')
  exp_key_str = 'global_{}'.format(time_stamp)
  output_dir = os.path.join(args.output_dir, exp_key_str)
  ckpt_dir = args.ckpt_dir
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(ckpt_dir, exist_ok=True)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
  device = torch.device(args.device)

  # Get DataLoader
  loader_dict = get_loader_dict(
      stage='global', 
      data_root=args.data_root,
      num_workers=args.workers,
      gs_img_h=args.gs_img_h,
      gs_img_w=args.gs_img_w,
      test_list=['train', 'test1', 'test2'],
      batch_sz_list=[args.batch_size, 1, 1],
  )

  # Model and trainable parameters.
  model = TVSegModel(
    tv_model=args.model_name, 
    aux_loss=args.aux_loss, 
    in_ch=3, 
    num_classes=19
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
  
  # Train and validation
  best_val_stat = {}
  for epoch in range(args.start_epoch, args.epochs):

    train_one_epoch(
      epoch=epoch, 
      model=model, 
      device=device, 
      data_loader=loader_dict['train'], 
      optimizer=optimizer, 
      lr_scheduler=lr_scheduler,
      aug_freq=args.aug_freq)

    test_dict = validation(
      epoch=epoch, 
      model=model, 
      device=device, 
      test_list=['test1', 'test2'], 
      loader_dict=loader_dict,
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
          'best_All.pth'
        )
      print('Epoch={}, Dataset={}, Aug_Freq={}, MRE={:.4f}'.format(
        epoch, args.data_root, args.aug_freq, best_val_mre))

    # Save training model
    if epoch % args.save_freq == 0:
      save_model(model, optimizer, lr_scheduler, epoch, ckpt_dir,
        'All.pth'.format(epoch)
      )

    # Logging
    log_gs(
      epoch=epoch, 
      output_dir=output_dir,
      best_val_stat=best_val_stat, 
      test_dict=test_dict,
      test_list=['test1', 'test2'],
    )


def parse_args():
  parser = argparse.ArgumentParser(description='Peacock Training')

  # Environmental settings
  parser.add_argument('--gpu_num', default='0', type=str)
  parser.add_argument('--device', default='cuda', help='device')
  parser.add_argument('--workers', default=16, type=int)
  parser.add_argument('--ckpt_dir', default='./ckpt/global')
  parser.add_argument('--output_dir', default='./output')
  parser.add_argument('--data_root', default='./data', type=str)
  
  # Model
  parser.add_argument('--model_name', default='deeplabv3_resnet50')
  parser.add_argument('--aux_loss', action='store_false')
  parser.add_argument('--save_freq', default=30, type=int)

  # Augmentation
  parser.add_argument('--aug_freq', default=0.1, type=float) 
  parser.add_argument('--batch_size', default=2, type=int)
  
  # Training
  parser.add_argument('--gs_img_h', default=720, type=int)
  parser.add_argument('--gs_img_w', default=580, type=int)
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('--epochs', default=500, type=int)
  parser.add_argument('--lr', default=0.0001, type=float)
  parser.add_argument('--weight_decay', default=0, type=float)

  # Resume training
  parser.add_argument('--resume', default='', action='store_false')

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = parse_args()
  main(args)
