#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:52:08 2018

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import cv2

import caffe
import h5py

import plots

import sklearn
import sklearn.datasets
import sklearn.linear_model


# ------------------------------------------
# Vectors
idCar = []; # id com todos os carros
idModel = []; # id com todos os modelos sao 250 diferentes

idRe = []; # id reidentificacao
filenames = []; # filenames

X = []; # processing data
y = []; # label data, preference for 'reid'

# ------------------------------------------
# Configurations
rW = 30; # resize width
rH = 30; # resize height

train_filename = '../data/train.h5';
test_filename = '../data/test.h5';
processingHDF5 = False;

# ------------------------------------------
# Temporaries
length = 0; # length for some vector
last_progress = 0;

file = open('../../../data/VehicleID_V1.0/attribute/model_attr.txt', 'r');
data = file.read();
data = data.split('\n');

# Read and save metadata about models
for row in data:
    if(row!=''):
        idCar.append(int(row.split(' ')[1]));
        idModel.append(int(row.split(' ')[0]));
file.close()

# fig, axs = plt.subplots(1, 2, tight_layout=True)
#fig, axs = plt.subplots(1, tight_layout=True)
#plots.barplot(idCar);

# Read and save metadata about re-identification
file = open('../data/img2vid.txt', 'r');
data = file.read();
data = data.split('\n')
length = len(data); i = 0;

for row in data:
    if(row!=''):
        filenames.append(row.split(' ')[0]);
        idRe.append(int(row.split(' ')[1]));
        
#        if(i/length>last_progress+0.01):
#            print('Loaded metadata '+str(last_progress*100))
#            last_progress = i/length;
#        i += 1;
file.close()

# ------------------------------------------------------------------------
# -----------------------  Load and save  --------------------------------
# ------------------------------------------------------------------------

# Read each image, resize and write in HDF5
if(processingHDF5):        
    # ------------------------------------------
    # Write data in H5F
    length = len(filenames);
    print('Initializing load images...')
    for i in range(length):
        X.append( cv2.resize( caffe.io.load_image('../../../data/VehicleID_V1.0/image/'+filenames[i]+'.jpg'), (rW, rH) ))
        y.append(idRe[i]);
        
        if(i/length>last_progress+0.01):
            print('Loaded '+str(last_progress*100))
            last_progress = i/length;

    # After loaded treat dataset
    print('Separete train and test data')
    Xtrain, Xtest, Ytrain, Ytest = sklearn.model_selection.train_test_split(X, y)
    
    Xtrain = np.array(Xtrain).transpose(0,3,1,2);
    Xtest = np.array(Xtest).transpose(0,3,1,2);
    
    Ytrain = np.array(Ytrain);
    Ytest = np.array(Ytest);
    
    print('Writing hdf5 files')
    with h5py.File(train_filename, 'w') as f:
        f['data'] = Xtrain;
        f['label'] = Ytrain;
    
    with h5py.File(test_filename, 'w') as f:
        f['data'] = Xtest;
        f['label'] = Ytest;

else:
    print('Loading hdf5 files')
    # Train
    h5data = h5py.File(train_filename, 'r');
    Xtrain = h5data['data'].value; # data
    Ytrain = h5data['label'].value; # labels
    
    # Test
    h5datat = h5py.File(test_filename, 'r');
    Xtest = h5datat['data'].value; # data
    Ytest = h5datat['label'].value; # labels

# ------------------------------------------------------------------------
# --------------------------  Solvers  -----------------------------------
# ------------------------------------------------------------------------

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.get_solver('../models/solver.prototxt')

#solver = caffe.AdamSolver('models/solver.prototxt')
#solver = None
#solver = caffe.get_solver('models/solver.prototxt')
#
##solver=caffe.get_solver('prototxtfile.prototxt')
##solver.net.copy_from('weights.caffemodel')
#
#
#blob = caffe.proto.caffe_pb2.BlobProto()
#data = open( './metadata/VGG_mean.binaryproto', 'rb' ).read()
#
#blob.ParseFromString(data)
#arr = np.array( caffe.io.blobproto_to_array(blob) )
#out = arr[0]

begin = time.time()
### solve
niter = 1000;  # EDIT HERE increase to train for longer
test_interval = niter / 100;
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

estimated = [];
original = [];
incorrect = 0;

# the main solver loop
for it in range(niter):
    print('Iteration: {0:d} of {1:d}, time elapsed {2:.2f}'.format(it, niter, time.time() - begin))
    solver.step(1);  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data;
    
    # run a full test every so often
    if it % test_interval == 0:
        print('Iteration '+str(it)+' testing...')
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            #correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
            incorrect += sum(solver.test_nets[0].blobs['negative'].data==solver.test_nets[0].blobs['anchor'].data);
            
            estimated.append(solver.test_nets[0].blobs['negative'].data.argmax(1).copy())
            original.append(solver.test_nets[0].blobs['anchor'].data.copy())
        # computa os acertos
        #test_acc[int(it // test_interval)] = correct / 1e4
#solver.step(0)
#_, ax1 = subplots()
#ax2 = ax1.twinx()
#ax1.plot(arange(niter), train_loss)
#ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
#ax1.set_xlabel('iteration');
#ax1.set_ylabel('train loss');
#ax2.set_ylabel('test accuracy');
#ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]));

elapsed = time.time() - begin;
print('Elpased time: '+str(elapsed))