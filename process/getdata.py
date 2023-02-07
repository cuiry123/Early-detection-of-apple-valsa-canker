from keras.utils import np_utils
from scipy.io import loadmat
import numpy as np
import os

# 获取类别文件夹及类别数
def get_dir(path):
    file_list = os.listdir(path)
    file_num = len(path)

    return file_list, file_num

# 获取所有.mat文件路径和标签，生成list
def path_label_acquisition(file_path, len_dir):

    path_name_list = os.listdir(file_path)    #获取.mat文件名称
    file_num = len(path_name_list)            #获取.mat文件数量

    path_list = []
    label_list = []
    for i in range(file_num):
        path = file_path + path_name_list[i]
        path_list.append(path)
        label_name = file_path[len_dir:len(file_path)-1]
        if label_name == 'health':
            label = 1     #健康
        else:
            label = 0     #患病

        label_list.append(label)

    return path_list, label_list, file_num

# 读取数据集中数据及标签
def get_data(dir_path, classes):

    class_name_list, class_num = get_dir(dir_path)
    len_dir = len(dir_path)
    data_set = []  # 数据列表
    Y = []  #标签列表

    for file in class_name_list:
        file_path = dir_path + file + '/'
        path_list, label_list, file_num = path_label_acquisition(file_path, len_dir)

        for i in range(file_num):
            path = path_list[i]
            data = loadmat(path)  # 获取数据字典
            keys = list(data.keys())
            features = data[keys[3]]  # 获取数据
            #feature_shape = features.shape
            data_set.append(features)  # 获取数据列表
            label = label_list[i]
            Y.append(label)

    data_set = np.array(data_set)
    data_set = data_set[:, :, :, :, np.newaxis]  # 增加维度

    Y = np.array(Y)
    Y = np_utils.to_categorical(Y, classes)  #one-hot

    return data_set, Y