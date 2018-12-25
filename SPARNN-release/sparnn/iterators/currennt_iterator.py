__author__ = 'sxjscience'

from netCDF4 import Dataset

from sparnn.utils import *
from sparnn.iterators import DataIterator

logger = logging.getLogger(__name__)


class CurrenntIterator(DataIterator):
    def __init__(self, iterator_param):
        super(CurrenntIterator, self).__init__(iterator_param)
        self.load(self.path)

    def load(self, path):
        self.data = {}
        ncFile = Dataset(path, 'r')
        seqLengths = ncFile.variables['seqLengths'][:]
        inputs = ncFile.variables['inputs'][:]
        outputs = ncFile.variables['targetPatterns'][:]
        self.data['dims'] = numpy.asarray(
            [[len(ncFile.dimensions['inputPattSize']), 1, 1], [len(ncFile.dimensions['targetPattSize']), 1, 1]],
            dtype="int32")
        self.data['input_raw_data'] = inputs.reshape((inputs.shape[0], inputs.shape[1], 1, 1))
        self.data['output_raw_data'] = outputs.reshape((outputs.shape[0], outputs.shape[1], 1, 1))
        self.data['clips'] = numpy.zeros((2, len(seqLengths), 2), dtype="int32")
        pos = 0
        for i in range(len(seqLengths)):
            self.data['clips'][0, i, 0] = pos
            self.data['clips'][0, i, 1] = seqLengths[i]
            self.data['clips'][1, i, 0] = pos
            self.data['clips'][1, i, 1] = seqLengths[i]
            pos += seqLengths[i]
        self.check_data()