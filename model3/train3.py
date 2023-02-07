import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.callbacks import TensorBoard
import time
import matplotlib.pylab as plt
import numpy as np
import os
import sys
from cnn_model3 import cnn_model3
sys.path.append("../process")#添加其他文件夹
from getdata import get_data
from getspedata  import get_spedata

def train(model, X_train, X_spetrain, Y_train, X_val, X_speval, Y_val, batch_size, epochs):
    # 模型编译
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(0.0003),    # 梯度优化方法
        loss=tf.keras.losses.categorical_crossentropy,    # 损失函数
        metrics=[tf.keras.metrics.categorical_accuracy])  #精度

    # 模型参数保存路径
    model_dir = '../weights/3D_CNN3_SNV/'
    model_file = 'model_weights'
    model_saved_path = model_dir + '/' + model_file

    # 是否加载已训练模型
    is_load_model = input('Would you like load the existed model weights if it exist ? y/n\n')
    if is_load_model == 'y':
        model.load_weights(model_saved_path)   #加载权重
        print('An existed model_weight table has been gotten...')
    else:
        print('A new model will be gotten...')

    #epoch = int(input('Please input epoch : '))
    #if epoch < 0:
    #    epoch = 0
    #print('*****************************************')

    model_name = "Model3SNV"
    tensorboard = TensorBoard(log_dir='../logs/{}'.format(model_name))

    # 模型训练
    hist = model.fit(
        [X_train,X_spetrain],
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True, # 在每个epoch之前对训练数据进行洗牌
        validation_data = ([X_val,X_speval], Y_val),
        callbacks = [tensorboard]
    )

    # 模型保存
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model.save(model_saved_path)
    model.save_weights(model_saved_path)
    print('model_weights have been saved at ' + model_saved_path)
    #model.summary()

    # 查看结果图
    plt.style.use("ggplot")   #matplotlib的美化样式
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N),hist.history["loss"],label ="train_loss")   #model的history有四个属性，loss,val_loss,acc,val_acc
    plt.plot(np.arange(0,N),hist.history["val_loss"],label="val_loss")
    plt.plot(np.arange(0,N),hist.history["categorical_accuracy"],label="train_acc")
    plt.plot(np.arange(0,N),hist.history["val_categorical_accuracy"],label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("../result/3/result_SNV.png")
    plt.show()


if __name__ =="__main__":
    channels = 1
    height = 80
    width = 130
    depth = 480
    class_num = 2
    batch_size = 2
    epochs = 100
    train_path = '../data/train/'
    val_path = '../data/val/'
    spe_train_path = '../spedata/train/'
    spe_val_path = '../spedata/val/'
    model = cnn_model3(channel=channels, height=height, width=width, depth=depth, classes=class_num) #网络

    X_train, Y_train = get_data(train_path, class_num)  #获取训练集数据
    X_val, Y_val = get_data(val_path, class_num)        #获取验证集数据
    X_spetrain ,Y1 = get_spedata(spe_train_path,class_num)
    X_speval, Y2 = get_spedata(spe_val_path, class_num)

    print('Train shape: ', X_train.shape, 'Validation shape: ', X_val.shape)
    print('Spe Train shape: ', X_spetrain.shape, 'Spe Validation shape: ', X_speval.shape)

    train(model, X_train, X_spetrain, Y_train, X_val, X_speval, Y_val, batch_size, epochs)   #训练