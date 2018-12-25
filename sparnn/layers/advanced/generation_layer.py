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

'''

Generation Layer is the layer for generating words/sequences

inner_layer_param is a list of dict that contains the required parameters for


'''

# TODO Here I think the implementation should be revised using acyclic graph
class GenerationLayer(Layer):
    def __init__(self, layer_param):
        super(GenerationLayer, self).__init__(layer_param)
        assert self.input.ndim == 4
        self.inner_layers = layer_param['inner_layers']
        self.n_steps = layer_param['n_steps']
        self.states = None
        self.fprop()

    def set_name(self):
        self.name = "GenerationLayer-" + str(self.id)

    '''
        In the GenerationLayer.__step_prediction function, use *args to handle multiple input
    '''

    def step_fprop(self, x_tm1, *args):
        # First Check if the number of variables is the same as our expectation
        assert len(args) == sum(len(layer.init_states()) if layer.is_recurrent else 0 for layer in self.inner_layers)
        last_output = x_tm1
        pos = 0
        states = ()
        for layer in self.inner_layers:
            if layer.is_recurrent:
                state_num = len(layer.init_states())
                if "ConvLSTMLayer" in type(layer).__name__:
                    hidden_output, cell_output = layer.step_fprop(last_output, None, *args[pos:pos + state_num])
                    states += (hidden_output, cell_output)
                elif "ConvRNNLayer" in type(layer).__name__:
                    hidden_output = layer.step_fprop(last_output, None, *args[pos:pos + state_num])
                    states += (hidden_output, )
                else:
                    assert False
                last_output = hidden_output
                pos += state_num
            else:
                last_output = layer.step_fprop(last_output)
        last_output = TT.unbroadcast(last_output, *range(last_output.ndim))
        ret = (last_output,) + states
        return ret

    def fprop(self):
        init_states = []
        for layer in self.inner_layers:
            init_states += list(layer.init_states()) if layer.is_recurrent else []
        scan_output, self.output_update = quick_scan(fn=self.step_fprop,
                                                     outputs_info=[self.input] + init_states,
                                                     n_steps=self.n_steps)
        self.output = TT.concatenate([TT.shape_padleft(self.input, 1), scan_output[0]])
        self.states = scan_output[1:]

    def print_stat(self):
        logger.info(self.name + ":")
        logger.info("   Total " + str(self.n_steps) + " steps")
        logger.info("   Generated Using:")
        for layer in self.inner_layers:
            logger.info("      " + layer.name)
