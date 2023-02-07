from keras.utils import np_utils
from scipy.io import loadmat
import numpy as np
import os

def get_dir(path):
    file_list = os.listdir(path)
    file_num = len(path)
    return file_list, file_num

def spe_acquisition(spe_path,len_dir):

    path_name_list = os.listdir(spe_path)
    spe_num = len(path_name_list)

    spe_path_list = []
    label_list = []
    for i in range(spe_num):
        path = spe_path + path_name_list[i]
        spe_path_list.append(path)
        label_name = spe_path[len_dir:len(spe_path)-1]
        if label_name == 'health':
            label = 1     #健康
        else:
            label = 0     #患病

        label_list.append(label)

    return spe_path_list, label_list, spe_num

def get_spedata(dir_path, classes):

    class_name_list, class_num = get_dir(dir_path)
    len_dir = len(dir_path)
    spedata_set = []  # 数据列表
    Y = []  #标签列表

    for file in class_name_list:
        file_path = dir_path + file + '/'
        spe_path_list, label_list, spe_num = spe_acquisition(file_path,len_dir)

        for i in range(spe_num):
            path = spe_path_list[i]
            data = loadmat(path)  # 获取数据字典
            keys = list(data.keys())
            features = data[keys[3]].T # 获取数据
            # feature_shape = features.shape
            spedata_set.append(features)  # 获取数据列表
            label = label_list[i]
            Y.append(label)

    spedata_set = np.array(spedata_set)

    Y = np.array(Y)
    Y = np_utils.to_categorical(Y, classes)  #one-hot

    return spedata_set, Y