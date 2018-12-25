__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
from sparnn.utils import *
from sparnn.layers import Layer
from sparnn.layers import ConvLSTMLayer
from sparnn.layers import ConvRNNLayer

logger = logging.getLogger(__name__)


# TODO (by sxjscience) I think bidirectional layer can be further extended to multi-dimensional case.
# The current implementation is to concatenate two LSTM/RNN layers
# Also, in my opinion, the current implementation is not so beautiful, should be revised later
class BidirectionalLayer(Layer):
    def __init__(self, layer_param):
        super(BidirectionalLayer, self).__init__(layer_param)
        self.inner_layer_type = layer_param['inner_layer_type']
        self.__init_inner_layer(layer_param)
        self.param = self.forward_layer.param + self.backward_layer.param

    def set_name(self):
        self.name = "BidirectionalLayer-" + str(self.id)

    def __init_inner_layer(self, param):
        assert self.dim_out[0] > 1
        layer_param = param.copy()
        if self.inner_layer_type == "ConvLSTM":
            layer_param['dim_out'] = (self.dim_out[0]/2, self.dim_out[1], self.dim_out[2])
            layer_param['input'] = self.input
            layer_param['mask'] = self.mask
            layer_param['id'] = self.id + ".Forward"
            self.forward_layer = ConvLSTMLayer(layer_param)
            layer_param['dim_out'] = (self.dim_out[0]/2 + self.dim_out[0]%2, self.dim_out[1], self.dim_out[2])
            layer_param['input'] = self.input[::-1, :, :, :, :]
            if self.mask is not None:
                layer_param['mask'] = self.mask[::-1, :]
            layer_param['id'] = self.id + ".Backward"
            self.backward_layer = ConvLSTMLayer(layer_param)
            self.output = TT.concatenate([self.forward_layer.output, self.backward_layer.output[::-1, :, :, :, :]], axis=2)
            self.cell_output = TT.concatenate([self.forward_layer.cell_output, self.backward_layer.cell_output[::-1, :, :, :, :]], axis=2)
        elif self.inner_layer_type == "ConvRNN":
            layer_param['dim_out'] = (self.dim_out[0]/2, self.dim_out[1], self.dim_out[2])
            layer_param['input'] = self.input
            layer_param['mask'] = self.mask
            layer_param['id'] = self.id + ".Forward"
            self.forward_layer = ConvRNNLayer(layer_param)
            layer_param['dim_out'] = (self.dim_out[0]/2 + self.dim_out[0]%2, self.dim_out[1], self.dim_out[2])
            layer_param['input'] = self.input[::-1, :]
            if self.mask is not None:
                layer_param['mask'] = self.mask[::-1, :]
            layer_param['id'] = self.id + ".Backward"
            self.backward_layer = ConvRNNLayer(layer_param)
            self.output = TT.concatenate([self.forward_layer.output, self.backward_layer.output[::-1,:]], axis=2)
        else:
            assert False








