# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#

# Import global packages
import os
import scipy.io as sio
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import SequentialSampler, DataLoader

# Import global constants
from constants import DET_BDD_LIST

# Import user defined classes and functions
from isbi_dataset import ISBIDataSet


def get_loader_dict(
  data_root, 
  num_workers, 
  gs_img_h, 
  gs_img_w, 
  stage, 
  test_list, 
  batch_sz_list):
  """ Make dictionary of torchvision.utils.data.Dataloader
  Args:
    args (argparse): Arguments for experiments
    stage (str): Stage names ('global' / 'local')
    test_list (list): List of test names (str)

  Returns:
    loader_dict (dictionary): Dictionary of Dataloaders
      Shape = {'test1': Dataloader,
               'test2': Dataloader,
               ...}      
  """
  loader_dict = {}
  
  if stage == 'global': # Global stage datasets: with downsampling
    
    transforms=None 
    img_h=gs_img_h
    img_w=gs_img_w

  elif stage == 'local': # Local stage datasets: without downsampling
    
    transforms=ToTensor()
    img_h=0
    img_w=0
  
  # Write loader dictionary
  for t_idx, t_name in enumerate(test_list):

    ds = ISBIDataSet(
      mode=t_name, 
      data_root=data_root, 
      transforms=transforms, 
      img_h=img_h, 
      img_w=img_w
    )    

    loader = DataLoader(
      ds, 
      batch_size=batch_sz_list[t_idx], 
      sampler=SequentialSampler(ds), 
      num_workers=num_workers
    )

    loader_dict[t_name] = loader

  return loader_dict


def write_result(exp_str, args, output_dir, test_list, stat_dict):

  os.makedirs(output_dir, exist_ok=True)
  
  def write_stat_dict(stat_file_name, stat_dict):
    with open(stat_file_name, 'a+') as stat_file:
      stat_file.write('{}\n'.format(exp_str))
      for key, _ in stat_dict.items():
        stat_file.write('{}\t'.format(key))
      stat_file.write('\n')
      for _, value in stat_dict.items():
        stat_file.write('{:.4f}\t'.format(value))
      stat_file.write('\n')

  def write_mre_sdr(eval_str, mre, sdr):
      mre_file_name = os.path.join(output_dir, '{}_mre.txt'.format(eval_str))
      write_stat_dict(mre_file_name, mre)

      for det_bdd in DET_BDD_LIST:
        sdr_file_name = os.path.join(output_dir,
          '{}_sdr{}.txt'.format(eval_str, det_bdd)
        )
        write_stat_dict(sdr_file_name, sdr[det_bdd])

  # Write stat for tests
  for t_name in test_list:
    eval_str = t_name
    mre = stat_dict[t_name]['MRE']
    sdr = stat_dict[t_name]['SDR']
    write_mre_sdr(eval_str, mre, sdr)

  # Write summary data
  summary_file_name = os.path.join(output_dir, '{}_summary.txt'.format(exp_str))
  with open(summary_file_name, 'a+') as summary_file:

    summary_file.write(str(args))
    summary_file.write('\n')

    # Write stat for tests
    for t_name in test_list:
      mre = stat_dict[t_name]['MRE']
      sdr = stat_dict[t_name]['SDR']
      scr = stat_dict[t_name]['SCR']

      summary_file.write(
        '{}\t{:04f}\t{:04f}\t{:04f}\t{:04f}\t{:04f}\t{}'.format(
        t_name, mre['average'], sdr[20]['average'], sdr[25]['average'], 
        sdr[30]['average'], sdr[40]['average'],
        '\t'.join(map(str, scr.numpy()))
        )
      )
      summary_file.write('\n')


def save_as_mat(mat_data, mat_name):
  """ Save mat_data as mat

  Args:
    output_dir (str): Output dir
    mat_data (dict): Data to save as mat file
    mat_name (str): Name of mat file
  """
  sio.savemat('{}.mat'.format(mat_name),
    {'x':mat_data['x'].detach().clone().numpy(), 
     'y':mat_data['y'].detach().clone().numpy()}
  )

    
def save_model(model, optim, lr_s, epoch, ckpt_dir, name):
  torch.save(
    {
      'model': model.state_dict(),
      'optimizer': optim.state_dict(),
      'lr_scheduler': lr_s.state_dict(),
      'epoch': epoch,
    },
    os.path.join(ckpt_dir, name)
  )  


def print_stat(test_list, stat_dict):
  """
  """
  for t_name in test_list:
    print('| {}: MRE: {:.4f} SDR: {:.4f} SCR: {:.4f}'.format(    
      t_name,
      stat_dict[t_name]['MRE']['average'].cpu().numpy(),
      stat_dict[t_name]['SDR'][20]['average'].cpu().numpy(),
      stat_dict[t_name]['SCR'].cpu().mean(),
      )
    )