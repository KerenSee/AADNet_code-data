# load feature dataset
import os
import numpy as np

# import tensorflow as tf
def get_file(filename):
    ''' load time-frequency diagram '''
    #     os.remove(filename+'DS_Store')
    #     os.remove(filename+'.ipynb_checkpoints')
    dataTrain = list()
    labelTrain = list()
    for label in os.listdir(filename):
        for pic in os.listdir(filename + label):
            dataTrain.append(filename + label + '/' + pic)
            if 'Female' in label:
                labelTrain.append(0)
            if 'Male' in label:
                labelTrain.append(1)

    temp = np.array([dataTrain, labelTrain])
    temp = np.transpose(temp)
    np.random.shuffle(temp)
    image_list = temp[:, 0]
    label_list = temp[:, 1]
    label_list = [int(i) for i in label_list]
    return image_list, label_list

pathname = "Female_Male/"
image_list, label_list = get_file(pathname)

# read train data  figure-->tensor
# read_file
# decode_png
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

X = np.empty([12*3,32,32,3])
with tf.compat.v1.Session() as sess:
    for i in range(len(image_list)):
        # 读取图像文件
        image_raw_data = tf.compat.v1.gfile.GFile(image_list[i],'rb').read()
        # 将图像文件解码为tensor
        image_data = tf.image.decode_jpeg(image_raw_data)
#         print(image_data.shape)
        # 改变张量的形状 32*64
        resized = tf.compat.v1.image.resize_images(image_data, [32,32])
        resized = np.asarray(resized.eval(),dtype='uint8')      # asarray 生成数组
#         print(resized.shape)
        X[i,:,:,:]=resized

# train data 深度学习的图像一定要做归一
#tf.compat.v1.disable_eager_execution()

# tf.compat.v1.disable_v2_behavior()
X_Train = X[:,:,:,:]

Y = np.subtract(np.array(label_list), 0) # 2/1 转换成 1/0
Y = Y.reshape([12*3,1])
V = tf.compat.v1.one_hot(Y, depth=2,axis=1,dtype=tf.float32)
session = tf.compat.v1.Session()
Y = session.run(V)
session.close()
Y = Y.reshape([12*3,2])
X_Train = X_Train.astype('float32')

X_Train = X_Train/255.

print(Y.shape)
print(X_Train.shape)
# resize

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv1D, BatchNormalization, MaxPool2D
from keras.optimizers import Adam            # 优化器
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
import keras.backend as K
# from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler


input_shape = X_Train.shape[1:] #（64,64,4）
num_classes = 2
model_name = 'wavelet_cnn.h5'
#定义模型
model = Sequential()

model.add(Conv2D(16, (3,3), padding = 'same', input_shape = input_shape, kernel_initializer = 'he_normal', name = 'conv2d_1'))
model.add(Activation('relu',name = 'activation_1'))
model.add(MaxPool2D(pool_size=(4,4),name = 'maxpool2d_1'))
model.add(Dropout(0.5,name = 'dropout_1'))

model.add(Conv2D(32, (3,3), padding = 'same',kernel_initializer = 'he_normal', name = 'conv2d_2'))
model.add(Activation('relu',name = 'activation_2'))
model.add(MaxPool2D(pool_size=(2,2),name = 'maxpool2d_2'))
model.add(Dropout(0.5,name = 'dropout_2'))

model.add(Flatten(name = 'flatten_1'))
model.add(Dense(500,kernel_initializer='he_normal',name='dense_1'))
model.add(Activation('relu',name='activation_3'))
model.add(Dropout(0.3,name = 'dropout_3'))

model.add(Dense(num_classes, kernel_initializer='he_normal',name='dense_2'))
model.add(Activation('softmax',name='activation_4')) #outputs
model.summary()

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto')
''' 
def scheduler(epoch):
    if epoch % 30 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
'''

# 优化方法
# opt = keras.optimizers.Adam(lr = 0.0003,decay = 1e-6)
# model.compile(loss = 'binary_crossentropy',  metrics=['accuracy'])
# model.save(model_name)

# opt = keras.optimizers.Adam(lr = 0.001)
# 编译模型

# reduce_lr = LearningRateScheduler(scheduler)
# x_train,x_test, y_train, y_test = train_test_split(X_Train,Y,test_size=0.2, random_state=0)

#  normalization
# x_train = x_train/255
# x_test = x_test/255

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# history = model.fit(x_train, y_train, epochs=300, shuffle=False,validation_data=(x_test,y_test),batch_size=16)
# history = model.fit(X_Train, Y, shuffle=True, batch_size=32, epochs=100, verbose=1, validation_split=0.2)

from sklearn.utils import shuffle
X_Train,Y = shuffle(X_Train,Y, random_state=1280)

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

kfold = KFold(n_splits=12, random_state=45, shuffle=True)
scores = []
# kfold = StratifiedKFold(n_splits=6, shuffle=True,random_state=1)
# %matplotlib inline
import matplotlib.pyplot as plt
# Y_train = Y.argmax(1)

import matplotlib.pyplot as plt
# Y_train = Y.argmax(1)

opt = keras.optimizers.Adam(learning_rate=0.00025)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# opt = keras.optimizers.Adam(learning_rate=0.00025, weight_decay=1e-6)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

for train, test in kfold.split(X_Train, Y):
    h = model.fit(X_Train[train], Y[train], validation_data=(X_Train[test], Y[test]), epochs=50, batch_size=8)
    #  Visualization of training process
    scores.append([h.history["loss"], h.history["acc"],
                   h.history["val_loss"], h.history["val_acc"]])

scores = np.array(scores)
scores = np.mean(scores, axis=0)
# print(scores)
labs = ["loss", "acc", "val_loss", "val_acc"]
result = zip([l + '_mean' for l in labs], [s[-1] for s in scores])
[print(res) for res in result]
[plt.plot(scores[i], label=labs[i]) for i in range(len(scores))]
plt.legend()
plt.savefig('eval' + '.png', dpi=300, bbox_inches='tight')  # bbox_inches可完整显示
plt.show()
