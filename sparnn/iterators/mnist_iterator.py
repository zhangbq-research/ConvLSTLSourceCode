__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
from sparnn.utils import *
from sparnn.iterators import DataIterator
from sparnn.helpers import movingmnist

logger = logging.getLogger(__name__)

'''
This is a special Iterator for the Moving MNIST experiment.

'''


class MNISTIterator(DataIterator):
    def __init__(self, iterator_param):
        super(MNISTIterator, self).__init__(iterator_param)
        self.velocity_amplitude = iterator_param['velocity_amplitude']
        self.characternum = iterator_param['characternum']
        self.seqdim = iterator_param['seqdim']
        self.seq = numpy.zeros((self.seqdim[0], self.minibatch_size, 1, self.seqdim[1], self.seqdim[2]),
                               dtype='float32')
        self.images = numpy.load(self.path)['images']

    def begin(self, do_shuffle=True):
        self.generate()

    def no_batch_left(self):
        return True

    def total(self):
        return self.minibatch_size

    def input_batch(self):
        return [self.seq[:(self.seqdim[0] / 2), :, :, :, :]]

    def output_batch(self):
        return [self.seq[(self.seqdim[0] / 2):self.seqdim[0], :, :, :, :]]

    def next(self):
        if self.no_batch_left():
            return
        self.generate()

    def generate(self):
        seqnum = self.minibatch_size
        characternum = self.characternum
        mnist = self.images
        seqdim = self.seqdim
        character_indices = numpy.random.randint(mnist.shape[0], size=(seqnum, characternum))
        angles = numpy.random.random((seqnum, characternum)) * (2 * numpy.pi)
        magnitudes = numpy.random.random((seqnum, characternum)) * self.velocity_amplitude
        velocities = numpy.zeros((seqnum, characternum, 2), dtype='float32')
        velocities[:, :, 0] = magnitudes * numpy.cos(angles)
        velocities[:, :, 1] = magnitudes * numpy.sin(angles)
        xmin = float(mnist.shape[1]) / 2
        xmax = float(seqdim[1]) - float(mnist.shape[1]) / 2
        ymin = float(mnist.shape[2]) / 2
        ymax = float(seqdim[2]) - float(mnist.shape[2]) / 2
        positions = numpy.random.uniform(0.0 + mnist.shape[1] / 2, seqdim[1] - mnist.shape[1] / 2,
                                         (seqnum, characternum, 2))
        for i in range(seqnum):
            self.seq[:, i, 0, :, :] = movingmnist.generate_sequence(mnist[character_indices[i]], velocities[i, :, :],
                                                                    positions[i, :, :],
                                                                    seqdim, (xmin, xmax, ymin, ymax))

    def print_stat(self):
        super(MNISTIterator, self).print_stat()
        logger.info("   Character Num: " + str(self.characternum))
        logger.info("   Velocity Amplitude: " + str(self.velocity_amplitude))
        logger.info("   Seqdim: " + str(self.seqdim))
