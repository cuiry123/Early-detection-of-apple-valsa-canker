import os
import random
from shutil import copy2

# 获取文件夹中所有文件名称和文件数目
def getDir(filepath):
    pathlist = os.listdir(filepath)
    pathnum = len(pathlist)
    return pathlist, pathnum

# 创建训练集、验证集和测试集文件夹
def mkTotalDir(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    dic = ['train', 'val', 'test']
    for i in range(0, 3):
        dic_path = data_path + dic[i] + '/'
        class_name = ['disease', 'health']
        for j in range(0, 2):
            current_path = dic_path + class_name[j] + '/'
            # 这个函数用来判断当前路径是否存在，如果存在则不创建，如果不存在则可以成功创建
            isExists = os.path.exists(current_path)
            if not isExists:
                os.makedirs(current_path)
                print('successful ' + dic[i]+'/'+class_name[j])
            else:
                print(dic[i]+'/'+class_name[j]+'is existed')
    return

# 数据集划分 train:val:test = 8:1:1
def divide_Train_Val_Test(source_path, data_path):

    dir_name_list, dir_num = getDir(source_path)
    for dir_name in dir_name_list:
        origin_dir_path = source_path + dir_name + '/'

        file_name_list, file_num = getDir(origin_dir_path)
        random.shuffle(file_name_list)  # 打乱
        train_list = file_name_list[0:int(0.8 * file_num)]
        val_list = file_name_list[int(0.8 * file_num):int(0.9 * file_num)]
        test_list = file_name_list[int(0.9 * file_num):file_num]

        for train_file in train_list:
            origin_train_path = origin_dir_path + train_file
            new_train_path = data_path + 'train' + '/' + dir_name + '/' + train_file
            copy2(origin_train_path, new_train_path)
        for val_file in val_list:
            origin_val_path = origin_dir_path + val_file
            new_val_path = data_path + 'val' + '/' + dir_name + '/' + val_file
            copy2(origin_val_path, new_val_path)
        for test_file in test_list:
            origin_test_path = origin_dir_path + test_file
            new_test_path = data_path + 'test' + '/' + dir_name + '/' + test_file
            copy2(origin_test_path, new_test_path)

if __name__ == '__main__':
    # 不用提前建文件夹
    source_path = 'F:/data/4pca/data/'  # 原始数据存放路径
    data_path = '../data2/'  # 数据集路径
    mkTotalDir(data_path)
    divide_Train_Val_Test(source_path, data_path)
