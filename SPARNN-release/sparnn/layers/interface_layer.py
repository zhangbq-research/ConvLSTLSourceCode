__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
from sparnn.utils import *

logger = logging.getLogger(__name__)


class InterfaceLayer(object):
    def __init__(self, layer_param):
        self.id = layer_param['id']
        self.use_input_mask = layer_param.get('use_input_mask', False)
        self.use_output_mask = layer_param.get('use_output_mask', False)
        input_data_type = layer_param.get('input_data_type', theano.config.floatX)
        input_ndim = layer_param.get('input_ndim', 5)
        output_data_type = layer_param.get('output_data_type', theano.config.floatX)
        output_ndim = layer_param.get('output_ndim', 5)
        self.set_name()
        self.input = quick_symbolic_variable(ndim=input_ndim, name=self._s("input"), typ=input_data_type)
        self.output = quick_symbolic_variable(ndim=output_ndim, name=self._s("output"), typ=output_data_type)
        if self.use_input_mask:
            self.input_mask = quick_symbolic_variable(ndim=2, name=self._s("input_mask"), typ=theano.config.floatX)
        else:
            self.input_mask = None
        if self.use_output_mask:
            self.output_mask = quick_symbolic_variable(ndim=2, name=self._s("output_mask"), typ=theano.config.floatX)
        else:
            self.output_mask = None

    def set_name(self):
        self.name = "InterfaceLayer-" + str(self.id)

    def _s(self, s):
        return '%s.%s' % (self.name, s)

    def input_symbols(self):
        if self.use_input_mask:
            return [self.input, self.input_mask]
        else:
            return [self.input]

    def output_symbols(self):
        if self.use_output_mask:
            return [self.output, self.output_mask]
        else:
            return [self.output]

    def symbols(self):
        return self.input_symbols() + self.output_symbols()

    def print_stat(self):
        logger.info(self.name + ":")
        logger.info("   Use Input mask: " + str(self.use_input_mask))
        if self.use_input_mask:
            logger.debug("   Input Mask Type:" + str(self.input_mask.type) + " Mask Name: " + self.input_mask.name)
        logger.info("   Input Type: " + str(self.input.type) + " Input Name: " + self.input.name)
        logger.info("   Use Output mask: " + str(self.use_output_mask))
        if self.use_output_mask:
            logger.debug("   Output Mask Type:" + str(self.output_mask.type) + " Mask Name: " + self.output_mask.name)
        logger.info("   Output Type: " + str(self.output.type) + " Output Name: " + self.output.name)

