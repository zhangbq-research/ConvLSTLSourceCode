__author__ = 'sxjscience'

import logging

import numpy

from sparnn.helpers import gifmaker


logger = logging.getLogger(__name__)

'''
The Moving MNIST Generator, the original MNIST dataset is downloaded from
    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    mnist_pic.npy is the training images of the MNIST dataset

'''


def move_step(v0, p0, bounding_box):
    xmin, xmax, ymin, ymax = bounding_box
    assert (p0[0] >= xmin) and (p0[0] <= xmax) and (p0[1] >= ymin) and (p0[1] <= ymax)
    v = v0.copy()
    assert v[0] != 0.0 and v[1] != 0.0
    p = v0 + p0
    while (p[0] < xmin) or (p[0] > xmax) or (p[1] < ymin) or (p[1] > ymax):
        vx, vy = v
        x, y = p
        dist = numpy.zeros((4,))
        dist[0] = abs(x - xmin) if ymin <= (xmin - x) * vy / vx + y <= ymax else numpy.inf
        dist[1] = abs(x - xmax) if ymin <= (xmax - x) * vy / vx + y <= ymax else numpy.inf
        dist[2] = abs((y - ymin) * vx / vy) if xmin <= (ymin - y) * vx / vy + x <= xmax else numpy.inf
        dist[3] = abs((y - ymax) * vx / vy) if xmin <= (ymax - y) * vx / vy + x <= xmax else numpy.inf
        n = numpy.argmin(dist)
        if n == 0:
            v[0] = -v[0]
            p[0] = 2 * xmin - p[0]
        elif n == 1:
            v[0] = -v[0]
            p[0] = 2 * xmax - p[0]
        elif n == 2:
            v[1] = -v[1]
            p[1] = 2 * ymin - p[1]
        elif n == 3:
            v[1] = -v[1]
            p[1] = 2 * ymax - p[1]
        else:
            assert False
    return v, p


'''
bounding_box = (xmin, xmax, ymin, ymax)

'''


def generate_sequence(imgs, velocity, initial_pos, seqdim, bounding_box):
    seq = numpy.zeros(seqdim, dtype='float32')
    for img, v0, p0 in zip(imgs, velocity, initial_pos):
        v = v0
        p = p0
        for i in range(seqdim[0]):
            topleft_x = int(p[0] - img.shape[0] / 2)
            topleft_y = int(p[1] - img.shape[1] / 2)
            seq[i, topleft_x:topleft_x + 28, topleft_y:topleft_y + 28] = numpy.maximum.reduce([
                seq[i, topleft_x:topleft_x + 28, topleft_y:topleft_y + 28],
                img
            ])
            v, p = move_step(v, p, bounding_box)
    return seq


'''
All the frames of the sequence must be square

'''


def save_to_numpy_format(seq, input_seq_len, output_seq_len, path):
    assert 4 == seq.ndim
    assert input_seq_len + output_seq_len == seq.shape[1]
    dims = numpy.asarray([[1, seq.shape[2], seq.shape[3]]], dtype="int32")
    input_raw_data = seq.reshape((seq.shape[0] * seq.shape[1], 1, seq.shape[2], seq.shape[3]))
    clips = numpy.zeros((2, seq.shape[0], 2), dtype="int32")
    clips[0, :, 0] = range(0, input_raw_data.shape[0], seq.shape[1])
    clips[0, :, 1] = input_seq_len
    clips[1, :, 0] = range(input_seq_len, input_raw_data.shape[0] + input_seq_len, seq.shape[1])
    clips[1, :, 1] = output_seq_len
    numpy.savez_compressed(path, dims=dims, input_raw_data=input_raw_data, clips=clips)


def generator(seqnum=15000, characternum=3, seqdim=(20, 64, 64), dataset='sparnn/datasets/mnist_img.npz',
              savedir='data/moving-mnist-3-example'):
    print '... Loading MNIST Data'
    mnist = numpy.load(dataset)['images'][500:1000]
    print 'Complete Loading!!'
    print '... Generating Sequences'
    character_indices = numpy.random.randint(mnist.shape[0], size=(seqnum, characternum))
    angles = numpy.random.random((seqnum, characternum)) * (2 * numpy.pi)
    magnitudes = numpy.random.random((seqnum, characternum)) * 2.0 + 3.0
    velocities = numpy.zeros((seqnum, characternum, 2), dtype='float32')
    velocities[:, :, 0] = magnitudes * numpy.cos(angles)
    velocities[:, :, 1] = magnitudes * numpy.sin(angles)
    xmin = float(mnist.shape[1]) / 2
    xmax = float(seqdim[1]) - float(mnist.shape[1]) / 2
    ymin = float(mnist.shape[2]) / 2
    ymax = float(seqdim[2]) - float(mnist.shape[2]) / 2
    positions = numpy.random.uniform(0.0 + mnist.shape[1] / 2, seqdim[1] - mnist.shape[1] / 2,
                                     (seqnum, characternum, 2))
    seq = numpy.zeros((seqnum,) + seqdim, dtype='float32')
    for i in range(seqnum):
        seq[i, :, :, :] = generate_sequence(mnist[character_indices[i]], velocities[i, :, :], positions[i, :, :],
                                            seqdim, (xmin, xmax, ymin, ymax))
        if savedir is not '':
            if i < 20:
                gifmaker.save_gif(seq[i, :, :, :], savedir + "/" + str(i) + ".gif")
    if savedir is not '':
        save_to_numpy_format(seq[:10000], 10, 10, savedir + "/moving-mnist-train.npz")
        save_to_numpy_format(seq[10000:12000], 10, 10, savedir + "/moving-mnist-valid.npz")
        save_to_numpy_format(seq[12000:15000], 10, 10, savedir + "/moving-mnist-test.npz")
    else:
        return seq
