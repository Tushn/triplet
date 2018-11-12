import _init_paths
import config as cfg
from sampledata import sampledata
from utils.timer import Timer
import caffe
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver, output_dir, pretrained_model=None, gpu_id=0, data=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        
        self.data = data
        self.solver_text = solver

        print('solver: '+solver)
        self.solver = caffe.SGDSolver(solver)
        self.solver.net.layers[0].set_data(data)
#        f = open(solver, 'r')
#        self.solver = caffe.SGDSolver(f.read()) # .get_solver()
#        f.close()
        if pretrained_model is not None:
            print(('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model))
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        #self.solver.test_nets[0].blobs['data']
        self.solver.net.layers[0].set_data(data)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = (self.solver_param.snapshot_prefix.split('/')[-1] +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print('Wrote snapshot to: {:s}'.format(filename))

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        losstxt = os.path.join(self.output_dir, 'loss.txt')
        f = open(losstxt, 'w')

        while self.solver.iter < max_iters:

            timer.tic()
            self.solver.step(1)
            timer.toc()

            if self.solver.iter % (1 * self.solver_param.display) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time))
                print('time remains: {}s'.format(timer.remain(self.solver.iter, max_iters)))

            if self.solver.iter % cfg.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

            loss = self.solver.net.blobs['loss'].data[0]
            f.write('{} {}\n'.format(self.solver.iter - 1, loss))
            f.flush()

        f.close()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()


if __name__ == '__main__':
    """Train network."""
    solver = '../models/solver.prototxt' # 'models/solver.prototxt'
    output_dir = '../data/'
    pretrained_model = '../models/VGG_CNN_M_1024.caffemodel'
    gpu_id = 0
    data = sampledata()
    max_iters = cfg.MAX_ITERS

#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
    sw = SolverWrapper(solver, output_dir, pretrained_model, gpu_id, data)
#    sw = SolverWrapper(solver, output_dir, None, gpu_id, data)

    print('Solving...')
    sw.train_model(max_iters)
#    sw.solver.net.forward()
#    sw.solver.net.layers[0].data._sample[0]
#    blob = sw.solver.net.layers[0]._get_next_minibatch()
    print('done solving')


#myself = sw.solver.net.layers[0].getSelf()
#myself.set_data(data)
#sw.solver.net.layers[0]._data._sample