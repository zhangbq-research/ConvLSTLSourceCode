__author__ = 'chen'
#gimport h5py
import numpy
import os
import random
import scipy.io as sio
import commands
import glob
from sparnn.helpers import gifmaker
from PIL import Image


def run_baseline(test_file):
    assert test_file.endswith('.npz')
    data = numpy.load(test_file)

    index = 0
    for start_point in data['clips'][1, :, 0]:  # output, instance, startpoint
        test_case = data['input_raw_data'][start_point - 4: start_point + 20, 0, :, :]  # Timestep, FeatureDim, Row, Col
        test_case = test_case.swapaxes(0, 1).swapaxes(1, 2)
        sio.savemat('testcase.mat', {'seq': test_case, 'index': index})
        index += 1
        state, output = commands.getstatusoutput(
            '/Applications/MATLAB_R2014b.app/bin/matlab -nodesktop -nosplash -r demo')
        # print state
        # print output
        # break


def HKO_data_generator(
        directory_path,
        input_file_num,
        input_frame_num=10,
        output_frame_num=20,
        slot_size=40,
        channel=1,
        height=100,
        width=100,
        time_length_per_file=240):
    assert slot_size >= input_frame_num + output_frame_num

    # channel, height, width
    dims = numpy.asarray([channel, height, width], dtype=numpy.int32)

    test_instance_count = 0
    validation_instance_count = 0
    train_instance_count = 0
    slot_num_per_file = int(time_length_per_file / slot_size)
    instance_num_per_slot = slot_size - (input_frame_num + output_frame_num) + 1
    # Input/Output, Instance, StartPoint/Length
    test_clips = numpy.zeros([2, instance_num_per_slot * input_file_num, 2], dtype=numpy.int32)
    validation_clips = numpy.zeros([2, instance_num_per_slot * input_file_num, 2], dtype=numpy.int32)
    train_clips = numpy.zeros([2, instance_num_per_slot * (slot_num_per_file - 2) * input_file_num, 2],
                              dtype=numpy.int32)
    # Timestep, FeatureDim, Row, Col
    input_raw_data = numpy.zeros([time_length_per_file * input_file_num, channel, height, width], dtype=numpy.float32)

    file_index = 0
    file_list = os.listdir(directory_path)
    for i in range(len(file_list)):
        if not (file_list[i].endswith('.mat')):
            continue

        print directory_path + '/' + file_list[i]
        # 240 * 100 * 100, time * width * height
        mat = numpy.asarray(h5py.File(directory_path + '/' + file_list[i])['seq'][:], dtype=numpy.float32)  # type??
        mat /= 255
        assert time_length_per_file == mat.shape[0]
        assert width == mat.shape[1]
        assert height == mat.shape[2]

        file_start = time_length_per_file * file_index
        sample_slot_indexes = random.sample(range(slot_num_per_file), 2)
        print 'sample_slot_indexes:' + str(sample_slot_indexes) + '\n\n'
        for slot in range(slot_num_per_file):
            slot_start = slot * slot_size
            if slot == sample_slot_indexes[0]:
                for instance in range(instance_num_per_slot):
                    test_clips[0, test_instance_count, 0] = file_start + slot_start + instance  # start_point
                    test_clips[0, test_instance_count, 1] = input_frame_num  # length
                    test_clips[1, test_instance_count, 0] = file_start + slot_start + instance + input_frame_num
                    test_clips[1, test_instance_count, 1] = output_frame_num
                    test_instance_count += 1
            elif slot == sample_slot_indexes[1]:
                for instance in range(instance_num_per_slot):
                    validation_clips[
                        0, validation_instance_count, 0] = file_start + slot_start + instance  # start_point
                    validation_clips[0, validation_instance_count, 1] = input_frame_num  # length
                    validation_clips[
                        1, validation_instance_count, 0] = file_start + slot_start + instance + input_frame_num
                    validation_clips[1, validation_instance_count, 1] = output_frame_num
                    validation_instance_count += 1
            else:
                for instance in range(instance_num_per_slot):
                    train_clips[0, train_instance_count, 0] = file_start + slot_start + instance  # start_point
                    train_clips[0, train_instance_count, 1] = input_frame_num  # length
                    train_clips[1, train_instance_count, 0] = file_start + slot_start + instance + input_frame_num
                    train_clips[1, train_instance_count, 1] = output_frame_num
                    train_instance_count += 1

        ####### set raw data #######
        mat = numpy.asarray([mat])
        # chanel, time, w, h  ->  time, chanel, h, w
        mat = mat.swapaxes(0, 1).swapaxes(2, 3)
        input_raw_data[file_start: file_start + time_length_per_file, :, :, :] = mat
        file_index += 1

    print test_instance_count
    print validation_instance_count
    print train_instance_count
    numpy.savez_compressed('testdata', dims=dims, input_raw_data=input_raw_data, clips=test_clips)
    numpy.savez_compressed('validationdata', dims=dims, input_raw_data=input_raw_data, clips=validation_clips)
    numpy.savez_compressed('traindata', dims=dims, input_raw_data=input_raw_data, clips=train_clips)
    '''
    start_point = train_clips[0][instance_num_per_slot*5-1][0]
    end_point = train_clips[0][instance_num_per_slot*5-1][0] + train_clips[0][instance_num_per_slot*5-1][1]
    gifmaker.save_gif(input_raw_data[start_point:end_point, 0], 'input.gif')
    print start_point
    print end_point

    start_point = train_clips[1][instance_num_per_slot*5-1][0]
    end_point = train_clips[1][instance_num_per_slot*5-1][0] + train_clips[1][instance_num_per_slot*5-1][1]
    print start_point
    print end_point
    gifmaker.save_gif(input_raw_data[start_point:end_point, 0], 'output.gif')
    '''


def pixel_to_rainfall(img, a=118.239, b=1.5241):
    dBZ = img * 70.0 - 10.0
    dBR = (dBZ - 10.0 * numpy.log10(a)) / b
    R = numpy.power(10, dBR / 10.0)
    return R


'''
Function Name: skill_score

This function calculates several skill scores for the prediction, including

POD = hits / (hits + misses)
FAR = false_alarms / (hits + false_alarms)
CSI = hits / (hits + misses + false_alarms)
ETS = (hits - correct_negatives) / (hits + misses + false_alarms - correct_negatives)
correlation = (prediction*truth).sum()/(sqrt(square(prediction).sum()) * sqrt(square(truth).sum() + eps))


This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
and all inputs should be between 0~1

'''


def skill_score(prediction, truth, threshold=0.5):
    assert 3 == prediction.ndim
    assert 3 == truth.ndim
    bpred = (pixel_to_rainfall(prediction) > threshold)
    btruth = (pixel_to_rainfall(truth) > threshold)
    hits = numpy.logical_and((True == bpred), (True == btruth)).sum(axis=(1, 2))
    misses = numpy.logical_and((False == bpred), (True == btruth)).sum(axis=(1, 2))
    false_alarms = numpy.logical_and((True == bpred), (False == btruth)).sum(axis=(1, 2))
    correct_negatives = numpy.logical_and((False == bpred), (False == btruth)).sum(axis=(1, 2))
    eps = 1E-9
    POD = (hits + eps) / (hits + misses + eps).astype('float32')
    FAR = (false_alarms) / (hits + false_alarms + eps).astype('float32')
    CSI = (hits + eps) / (hits + misses + false_alarms + eps).astype('float32')

    correlation = (prediction * truth).sum(axis=(1, 2)) / (
        numpy.sqrt(numpy.square(prediction).sum(axis=(1, 2))) * numpy.sqrt(numpy.square(truth).sum(axis=(1, 2))) + eps)
    rain_rmse = numpy.square(pixel_to_rainfall(prediction) - pixel_to_rainfall(truth)).sum()
    rmse = numpy.square(prediction - truth).sum()
    return {"POD": POD, "FAR": FAR, "CSI": CSI, "correlation": correlation, "Rain RMSE": rain_rmse, "RMSE": rmse}


'''
Function Name: read_hko_mats

This function reads all results from the dirpath and calculates the average skill_score

'''


def baseline_score(dirpath, threshold=0.5):
    avg_score = {"POD": 0.0, "FAR": 0.0, "CSI": 0.0, "correlation": 0.0, "Rain RMSE": 0.0, "RMSE": 0.0}
    total_num = 0
    for filename in glob.glob(dirpath + "/*.mat"):
        content = sio.loadmat(filename)
        truth = content['seq'][:, :, 4:24]/255.0
        pred = content['predSeqNormal']/255.0
        score = skill_score(numpy.swapaxes(pred, 0, 2), numpy.swapaxes(truth, 0, 2), threshold)
        for key in score:
            avg_score[key] += score[key]
        total_num += 1
    print "Total Mat File in Path: ", dirpath, " is ", total_num
    avg_score = {key: avg_score[key]/total_num for key in avg_score}
    return avg_score


    # HKO_data_generator('/Users/chen/Downloads/', input_file_num = 2)
    # HKO_data_generator('/ghome/zchenbb/HKO100/', input_file_num = 97)
    # run_baseline('testdata.npz')