__author__ = 'sxjscience'

import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import NumpyIterator
from sparnn.layers import InterfaceLayer
from sparnn.layers import AggregatePoolingLayer
from sparnn.layers import DropoutLayer
from sparnn.layers import ConvLSTMLayer
from sparnn.layers import ConvForwardLayer
from sparnn.layers import ElementwiseCostLayer
from sparnn.layers import EmbeddingLayer
from sparnn.layers import GenerationLayer
from sparnn.models import Model

from sparnn.optimizers import SGD
from sparnn.optimizers import RMSProp
from sparnn.optimizers import AdaDelta

from sparnn.helpers import movingmnist

import os
import random
import numpy
import logging

random.seed(1000)
numpy.random.seed(1000)

save_path = "./moving-mnist/conv-unconditional-nopadding/4-5X5-5X5-5X5-128-64-64-mnist500/"
log_path = save_path + "mnist_sequence_forecasting_conv_unconditional.log"

if not os.path.exists(save_path):
    os.makedirs(save_path)

sparnn.utils.quick_logging_config(log_path)

iterator_param = {'path': 'data/moving-mnist-example/moving-mnist-train.npz', 'minibatch_size': 16,
                  'use_input_mask': False, 'input_data_type': 'float32', 'is_output_sequence': True,
                  'name': 'moving-mnist-train-iterator'}
train_iterator = NumpyIterator(iterator_param)
train_iterator.begin(do_shuffle=True)
train_iterator.print_stat()

iterator_param = {'path': 'data/moving-mnist-example/moving-mnist-valid.npz', 'minibatch_size': 16,
                  'use_input_mask': False, 'input_data_type': 'float32', 'is_output_sequence': True,
                  'name': 'moving-mnist-valid-iterator'}
valid_iterator = NumpyIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()

iterator_param = {'path': 'data/moving-mnist-example/moving-mnist-test.npz', 'minibatch_size': 16,
                  'use_input_mask': False, 'input_data_type': 'float32', 'is_output_sequence': True,
                  'name': 'moving-mnist-test-iterator'}
test_iterator = NumpyIterator(iterator_param)
test_iterator.begin(do_shuffle=False)
test_iterator.print_stat()

rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng()

input_seq_length = 10
output_seq_length = 10

param = {"id": "moving-mnist", "use_input_mask": False, "input_ndim": 5, "output_ndim": 5}
interface_layer = InterfaceLayer(param)

patch_size = 4
reshape_input = quick_reshape_patch(interface_layer.input, patch_size)
reshape_output = quick_reshape_patch(interface_layer.output, patch_size)
feature_num = patch_size*patch_size
row_num = int(64/patch_size)
col_num = int(64/patch_size)
data_dim = (feature_num, row_num, col_num)

logger.info("Data Dim:" + str(data_dim))
minibatch_size = interface_layer.input.shape[1]

middle_layers = []
param = {"id": 0, "rng": rng, "theano_rng": theano_rng, "dim_in": data_dim,
         "dim_out": (128, row_num, col_num), "input_receptive_field": (5, 5), "transition_receptive_field": (5, 5),
         "minibatch_size": minibatch_size,
#         "learn_padding": True,
         "input": reshape_input,
         "n_steps": input_seq_length}
middle_layers.append(ConvLSTMLayer(param))
param = {"id": 1, "rng": rng, "theano_rng": theano_rng, "dim_in": (128, row_num, col_num),
         "dim_out": (64, row_num, col_num), "input_receptive_field": (5, 5), "transition_receptive_field": (5, 5),
         "minibatch_size": minibatch_size,
#         "learn_padding": True,
#         "input_padding": middle_layers[0].hidden_padding,
         "input": middle_layers[0].output,
         "n_steps": input_seq_length}
middle_layers.append(ConvLSTMLayer(param))
param = {"id": 2, "rng": rng, "theano_rng": theano_rng, "dim_in": (64, row_num, col_num),
         "dim_out": (64, row_num, col_num), "input_receptive_field": (5, 5), "transition_receptive_field": (5, 5),
         "minibatch_size": minibatch_size,
#         "learn_padding": True,
#         "input_padding": middle_layers[1].hidden_padding,
         "input": middle_layers[1].output,
         "n_steps": input_seq_length}
middle_layers.append(ConvLSTMLayer(param))


param = {"id": 3, "rng": rng, "theano_rng": theano_rng, "dim_in": data_dim,
         "dim_out": (128, row_num, col_num), "input_receptive_field": (5, 5), "transition_receptive_field": (5, 5),
         "init_hidden_state": middle_layers[0].output[-1], "init_cell_state": middle_layers[0].cell_output[-1],
         "minibatch_size": minibatch_size,
#         "learn_padding": True,
         "input": None,
         "n_steps": output_seq_length - 1}
middle_layers.append(ConvLSTMLayer(param))
param = {"id": 4, "rng": rng, "theano_rng": theano_rng, "dim_in": (128, row_num, col_num),
         "dim_out": (64, row_num, col_num), "input_receptive_field": (5, 5), "transition_receptive_field": (5, 5),
         "init_hidden_state": middle_layers[1].output[-1], "init_cell_state": middle_layers[1].cell_output[-1],
         "minibatch_size": minibatch_size,
#         "learn_padding": True,
#         "input_padding": middle_layers[3].hidden_padding,
         "input": middle_layers[3].output,
         "n_steps": output_seq_length - 1}
middle_layers.append(ConvLSTMLayer(param))
param = {"id": 5, "rng": rng, "theano_rng": theano_rng, "dim_in": (64, row_num, col_num),
         "dim_out": (64, row_num, col_num), "input_receptive_field": (5, 5), "transition_receptive_field": (5, 5),
         "init_hidden_state": middle_layers[2].output[-1], "init_cell_state": middle_layers[2].cell_output[-1],
         "minibatch_size": minibatch_size,
#         "learn_padding": True,
#         "input_padding": middle_layers[4].hidden_padding,
         "input": middle_layers[4].output,
         "n_steps": output_seq_length - 1}
middle_layers.append(ConvLSTMLayer(param))


param = {"id": 6, "rng": rng, "theano_rng": theano_rng, "dim_in": (128 + 64 + 64, row_num, col_num),
         "dim_out": data_dim, "input_receptive_field": (1, 1),
         "input_stride": (1, 1), "activation": "sigmoid",
         "minibatch_size": minibatch_size,
         "conv_type": "same",
         "input": TT.concatenate([
             TT.concatenate([
                 middle_layers[0].output[-1:],
                 middle_layers[1].output[-1:],
		 middle_layers[2].output[-1:]], axis=2),
             TT.concatenate([
                 middle_layers[3].output,
                 middle_layers[4].output,
	         middle_layers[5].output], axis=2)])
         }
middle_layers.append(ConvForwardLayer(param))

param = {"id": "cost", "rng": rng, "theano_rng": theano_rng, "cost_func": "BinaryCrossEntropy",
         "dim_in": data_dim, "dim_out": (1, 1, 1),
         "minibatch_size": minibatch_size,
         "input": middle_layers[6].output,
         "target": reshape_output}
cost_layer = ElementwiseCostLayer(param)

outputs = [{"name": "prediction", "value": middle_layers[6].output}]

# error_layers = [cost_layer]

param = {'interface_layer': interface_layer, 'middle_layers': middle_layers, 'cost_layer': cost_layer,
         'outputs': outputs, 'errors': None,
         'name': "Moving-MNIST-Model-Convolutional-test-unconditional",
         'problem_type': "regression"}
model = Model(param)
model.print_stat()

param = {'id': '1', 'learning_rate': 1e-3, 'momentum': 0.9, 'decay_rate': 0.9, 'clip_threshold': None,
         'max_epoch': 200, 'start_epoch': 0, 'max_epochs_no_best': 200, 'decay_step': 200,
         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 30}
optimizer = RMSProp(model, train_iterator, valid_iterator, test_iterator, param)
optimizer.train()
