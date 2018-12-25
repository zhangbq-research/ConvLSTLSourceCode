import numpy
from PIL import Image, ImageSequence


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    # tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    # tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


'''
The visualization is

Assume the first dimension of weight_mat is the data_dimension and the second dimension of weight_mat is the feature_dimension
Also, all the data must be square!
'''


def visualize_weight(weight_mat, tile_shape=(10, 20), savepath=""):
    assert 2 == weight_mat.ndim
    data_dim = weight_mat.shape[0]
    feature_dim = weight_mat.shape[1]
    data_width = data_height = int(numpy.sqrt(data_dim))
    w = weight_mat[:, (weight_mat ** 2).sum(axis=0).argsort()[::-1][:(tile_shape[0] * tile_shape[1])]]
    output = tile_raster_images(w.transpose(), (data_height, data_width), tile_shape)
    if savepath is "":
        Image.fromarray(output).show()
    else:
        Image.fromarray(output).save(open(savepath, 'wb'))


def unfold_gif(gif_file, output='output.jpg', gap_width=15, step=1, direction='r'):
    assert 'r' == direction or 'c' == direction

    im = Image.open(gif_file)

    count = 0
    for frame in ImageSequence.Iterator(im):
        count += 1
        if count == 1:
            mode = frame.mode
            width, height = frame.size
    if 'r' == direction:
        out_im = Image.new(mode, ((width+gap_width)*(count/step)-gap_width, height), 255)
    else:
        out_im = Image.new(mode, (width, (height+gap_width)*(count/step)-gap_width), 255)
    im.close()
    im = Image.open(gif_file)
    count = 0
    index = 0
    l = []
    for frame in ImageSequence.Iterator(im):
        l.append(numpy.array(frame))
    for count in range(len(l)):
        if count % step == 0:
            if 'r' == direction:
                window = (index*(width+gap_width), 0, index*(width+gap_width)+width, height)  # (left, upper, right, lower)
            else:
                window = (0, index*(height+gap_width), width, index*(height+gap_width)+height)  # (left, upper, right, lower)
            out_im.paste(Image.fromarray(l[count]), window)
            index += 1
    out_im.save(output)


def stack_images(im_list, pad, output, direction='c'):
    assert(len(im_list) > 0)
    out_im = []
    for i in range(len(im_list)):
        im = Image.open(im_list[i])
        if 0 == i:
            width, height = im.size
            if 'c' == direction:
                out_im = Image.new(im.mode, (width, len(im_list)*(height+pad)-pad), 255)
            else:
                out_im = Image.new(im.mode, (len(im_list)*(width+pad)-pad, height), 255)
        if 'c' == direction:
            window = (0, i*(height+pad), width, i*(height+pad)+height)
        else:
            window = (i*(width+pad), 0, i*(width+pad)+width, height)
        out_im.paste(im, window)
    out_im.save(output)
