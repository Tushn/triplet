import caffe
import numpy as np
import cv2
import copy
import config as cfg
from sampledata import sampledata

from utils.blob import prep_im_for_blob, im_list_to_blob

import matplotlib.pyplot as plt

class DataLayer(caffe.Layer):
#    def __init__(self):
#        self.data = []
#        
    def getName(self):
        return "DataLayer";
    def getSelf(self):
        return self;
    """Sample data layer used for training."""

    def _shuffledata(self):
        print('Shuffling the data ...')
        sample = []
        sample_person = copy.deepcopy(self.data._sample_person)
#        sample_person = copy.deepcopy(self.top[0].data[...]._sample_person)
        
#        print('self.data: '+str(self.data))
        
        for i in sample_person.keys():
            np.random.shuffle(sample_person[i])
        while len(sample_person) > 0:
            person = np.random.choice(list(sample_person.keys()))
            while len(sample_person[person]) < cfg.CUT_SIZE:
                sample_person[person].append(
                    np.random.choice(self.data._sample_person[person]))
            num = 0
            while num < cfg.CUT_SIZE:
                sample.append(sample_person[person].pop())
                num += 1
            if len(sample_person[person]) == 0:
                sample_person.pop(person)
        self.data._sample = sample

    def _get_next_minibatch(self):
        # Sample to use for each image in this batch
        # Sample the samples randomly
        if self._index + self.batch_size <= len(self.data._sample):
            sample = self.data._sample[self._index:self._index + self.batch_size]
            self._index += self.batch_size
        else:
            sample = self.data._sample[self._index:]
            if cfg.TRIPLET_LOSS:
                self._shuffledata()
            else:
                np.random.shuffle(self.top[0].data[...]._sample)
            self._epoch += 1
            print('Epoch {}'.format(self._epoch))
            self._index = self.batch_size - len(sample)
            sample.extend(self.data._sample[:self._index])

        im_blob, labels_blob = self._get_image_blob(sample)
        plt.imshow(im_blob[0].transpose(1,2,0))
        blobs = {'data': im_blob, 'labels': labels_blob}
        return blobs

    def _get_image_blob(self, sample):
        im_blob = []
        labels_blob = []
        for i in range(self.batch_size):
#            im = cv2.imread(cfg.IMAGEPATH + sample[i]['picname'])
            im = cv2.imread('../../../../data' + sample[i]['picname'])
#            print('../../../../data' + sample[i]['picname'])
            if sample[i]['flipped']:
                im = im[:, ::-1, :]
            
            personname = sample[i]['picname'].split('/')[2]
            
            labels_blob.append(self.data._sample_label[personname])
            im = prep_im_for_blob(im)

            im_blob.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(im_blob)
        return blob, labels_blob

    def set_data(self, data):
        """Set the data to be used by this layer during training."""
        self.data = data
        if cfg.TRIPLET_LOSS:
            print('Epoch {}'.format(self._epoch))
            self._shuffledata()
        else:
            np.random.shuffle(self.data._sample)

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        print(' --- SETUP --- ')
        self.data = sampledata()
        if cfg.TRIPLET_LOSS:
            self.batch_size = cfg.TRIPLET_BATCH_SIZE
        else:
            self.batch_size = cfg.BATCH_SIZE
        self._name_to_top_map = {
            'data': 0,
            'labels': 1}

        self._index = 0
        self._epoch = 1
        
        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
#        top[0].reshape(self.batch_size, 3, 224, 224)
        top[0].reshape(self.batch_size, 3, 30, 30)
        top[1].reshape(self.batch_size)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.items():
            top_ind = self._name_to_top_map[blob_name]
            
#            print('top_ind: %s, blob_name: %s' % (top_ind, blob_name))
#            print('blob.shape: %s' % (str(blob.shape)))
            
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass