<img src="./images/mpa.svg" width="800" />

Multistage Probabilistic Approach for The Localization of Cephalometric Landmarks
======================
This page provides the source code of the paper 
'Multistage Probabilistic Approach for The Localization of Cephalometric Landmarks'  

## Requirements

### System
Our code requires the following environment.
  1. Ubuntu 18.04 LTS
  2. Cuda 10.0 or higher
  3. Python 3.6

### Database
We used the ISBI2015 dataset.
It is available at [ISBI2015](http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge1/) or
[here](https://drive.google.com/file/d/1eDIYn_cXtPy8RpR16sNpDM4murmvVa69/view?usp=sharing).

## Installation
You can install our code with the following steps.
1. Clone and extract our code. 
2. Install python packages with 'pip install -r requirements.txt'
3. Download ISBI2015 dataset and trained models with 'get_data_and_models.py'

## How to Test
Results in the paper can be reproduced with **'test.py'**.

## How to Train
The reproduction of results may require delicate calibrations of training parameters and system requirements.
1. Train global stage model with **'train_global.py'**.
2. Train local stage models with **'train_local.py'** for each landmark.
3. Train refinement stage models with **'train_refine.py'** for bilateral landmarks of the mandible (Gonion and Articulare).

## Acknowledgement
Our code is based on the PyTorch [[1](#ref-1)] and Kornia [[2](#ref-2)]. 
****
## References
<a name="ref-1"></a>[1] https://github.com/pytorch/pytorch \
<a name="ref-2"></a>[2] https://github.com/kornia/kornia\
