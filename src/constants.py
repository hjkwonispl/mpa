# -*- coding: utf-8 -*-
#
#  MPA Authors. All Rights Reserved.
#
""" Global constants for cephalometric landmarks and ISBI2015 dataset"""

# Number of Landmarks
NUM_LM = 19

# Size of training, test1, and test2 set
SZ_TRAINING = 150
SZ_TEST1 = 150
SZ_TEST2 = 100

# Raw image height, width, channel
RAW_IMG_H = 2400
RAW_IMG_W = 1935
RAW_IMG_C = 3

# Long Landmark Names
L_LM_NAME_DICT = { 0: 'Sella',
	1: 'Nasion',
	2: 'Orbitale',
	3: 'Porion',
	4: 'Subspinale',
	5: 'Supramentale',
	6: 'Pogonion',
	7: 'Menton',
	8: 'Gnathion',
	9: 'Gonion',
	10: 'Lower Incisal Incision',
	11: 'Upper Incisal Incision',
	12: 'Upper Lip',
	13: 'Lower Lip',
	14: 'Subnasale',
	15: 'Soft Tissue Pogonion',
	16: 'Posterior Nasal Spine',
	17: 'Anterior Nasal Spine',
	18: 'Articulate',
}

# Short Landmark Names
S_LM_NAME_DICT = {0: 'S',
	1: 'N',
	2: 'Or',
	3: 'Po',
	4: 'A',
	5: 'B',
	6: 'Pog',
	7: 'Me',
	8: 'Gn',
	9: 'Go',
	10: 'LI',
	11: 'UI',
	12: 'UL',
	13: 'LL',
	14: 'SN',
	15: 'sPog',
	16: 'PNS',
	17: 'ANS',
	18: 'Ar',
}

# Detection Boundaries
DET_BDD_LIST = [20, 25, 30, 40]

# Number of Measurement Methods for The Faical Type Classification
NUM_METHODS = 8

# Index of Gonion and Articulare
Go_IDX = 9
Ar_IDX = 18