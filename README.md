# Triplet

This code is a fork project from [Triplet of Peng Zhang](https://github.com/hizhangp/triplet). He implemented was triplet based [FaceNet](https://arxiv.org/abs/1503.03832v1). My version was change data for Caffe, how to install and use code is explained below.

## Installation

I recommend you install Anaconda and install Caffe by command [here](https://anaconda.org/conda-forge/caffe). You will need to download [caffemodel](http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_M_1024.caffemodel) and [database](http://vis-www.cs.umass.edu/lfw/). That caffemodel put in "models" directory and database in "faces", both directories in root this project.

## Executions

Run "train.py" in Anaconda, try to run foward: 

``` sw.solver.net.forward() ```

If are run, you can try use backward or ``` sw.train_model(max_iters) ```, for both forward and backward. In this point code not execute, because does not recognize ``` ._data ``` variable. But, you can print value with ```sw.net.layers[0]._data```.




