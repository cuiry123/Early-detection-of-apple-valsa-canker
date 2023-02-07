import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import load_model
import sys
sys.path.append("../process")
from getdata import get_data
from getspedata  import get_spedata

batch_size = 2
class_num = 2
test_path = '../data/test/'  #验证集数据存放处
spe_test_path = '../spedata/test/'

# 模型参数保存路径
model_dir = '../weights/3D_CNN3_SNV'
model_file = 'model_weights'
model_saved_path = model_dir + '/' + model_file

X_test, Y_test = get_data(test_path, class_num)  #获取测试集数据
X_spetest, Y = get_spedata(spe_test_path, class_num)

model = load_model(model_saved_path)  #加载模型

# 模型验证
loss, acc = model.evaluate(
    [X_test, X_spetest],
    Y_test,
    batch_size=batch_size
)

print('Test shape: ', X_test.shape)
print('Spe Test shape: ', X_spetest.shape)
print('Test loss: ', loss)
print('Test accuracy: ', acc)

#model.summary()