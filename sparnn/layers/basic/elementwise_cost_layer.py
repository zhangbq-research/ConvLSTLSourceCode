__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class ElementwiseCostLayer(Layer):
    def __init__(self, layer_param):
        super(ElementwiseCostLayer, self).__init__(layer_param)
        # TODO !Important In fact, when the input dimension of the cost layer is 4 means, you are actually mapping
        # TODO a sequence to a time-independent variable. Here I just add an if statement to do the decision
        self.target = layer_param['target']
        self.weight = layer_param.get('weight', None)
        self.cost_func = layer_param['cost_func']
        if self.weight is not None:
            assert (type(self.input) is list) and (type(self.target) is list) and (type(self.weight) is list) and (
                type(self.cost_func) is list) and (type(self.mask) is list)
            assert len(self.input) == len(self.target) == len(self.weight)
        self.fprop()

    def set_name(self):
        self.name = "ElementwiseCostLayer-" + str(self.id)

    def fprop(self):
        if self.weight is not None:
            self.output = sum(
                weight * quick_cost(input, target, cost_func, mask) for
                weight, target, input, cost_func, mask in
                zip(self.weight, self.target, self.input, self.cost_func, self.mask))
        else:
            self.output = quick_cost(self.input, self.target, self.cost_func, self.mask)

    def print_stat(self):
        logger.info(self.name + " : ")
        if self.weight is not None:
            logger.info("   Cost Function: " + str(self.cost_func))
            logger.info("   Weights: " + str(self.weight))
