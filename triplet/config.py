import os

# Training image data path
#IMAGEPATH = 'data/face/'
#IMAGEPATH = '../models/train.txt'
IMAGEPATH = '../../../../data/faces/'

# LFW image data path
LFW_IMAGEPATH = '../data/LFW/lfw-deepfunneled/'

# Path to caffe directory
#CAFFEPATH = '~/caffe'
CAFFEPATH = '~/anaconda3/bin'

# Snapshot iteration
SNAPSHOT_ITERS = 2000

# Max training iteration
MAX_ITERS = 500

# The number of samples in each minibatch for triplet loss
TRIPLET_BATCH_SIZE = 50

# The number of samples in each minibatch for other loss
BATCH_SIZE = 15

# If need to train tripletloss, set False when pre-train
TRIPLET_LOSS = True

# Use horizontally-flipped images during training?
FLIPPED = True

# training percentage
PERCENT = 0.8

# USE semi-hard negative mining during training?
SEMI_HARD = True

# Number of samples of each identity in a minibatch
CUT_SIZE = 5


TARGET_SIZE = 30