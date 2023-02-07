import os
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
    for i in range(3):
        dic_path = data_path + dic[i] + '/'
        class_name = ['disease', 'health']
        for j in range(2):
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
def divide_Train_Val_Test(spe_path, data_path, spedata_path):

    dic = ['train', 'val', 'test']
    class_name = ['disease', 'health']
    for i in range(3):
        data_dic_path = data_path + dic[i] + '/'
        for j in range(2):
            data_file_path = data_dic_path + class_name[j] + '/'
            file_name_list, file_num = getDir(data_file_path)
            for k in range(file_num):
                spe_name = file_name_list[k]
                oringin_path = spe_path + class_name[j] + '/' + spe_name
                new_path = spedata_path + dic[i] + '/' + class_name[j] + '/' + spe_name
                copy2(oringin_path, new_path)

if __name__ == '__main__':
    spe_path = 'E:/data/3prespe/msc/'  # 光谱数据原始存放路径
    data_path = '../data/'  # 图像数据集路径
    spedata_path = '../spedata/'  # 光谱数据集路径
    mkTotalDir(spedata_path)
    divide_Train_Val_Test(spe_path, data_path, spedata_path)