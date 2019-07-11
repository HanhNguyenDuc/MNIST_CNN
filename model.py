import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.callbacks import *


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
IMAGE_SIZE = X_train.shape[1:]


def Model_CNN():
    input_ = Input(shape = IMAGE_SIZE)
    conv_1a = Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(input_)   
    conv_1b = Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_1a)
    maxpool_1 = MaxPooling2D(pool_size = (2, 2))(conv_1b)
#     norm_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(maxpool_1)
    drop_1 = Dropout(0.25)(maxpool_1)
    
    
#     conv_2z = Conv2D(32, kernel_size = (1, 1), padding = 'same', activation = 'relu')(maxpool_1)
    conv_2a = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(drop_1)
    conv_2b = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_2a)
    maxpool_2 = MaxPooling2D(pool_size = (2, 2))(conv_2b)
    drop_2 = Dropout(0.25)(maxpool_2)
    
    conv_3a = Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu')(drop_2)
    conv_3b = Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu')(conv_3a)
    maxpool_3 = MaxPooling2D(pool_size = (2, 2))(conv_3b)
    drop_3 = Dropout(0.25)(maxpool_3)
    average_pooling = AveragePooling2D(pool_size = (4, 4))(drop_3)

    flatten_ = Flatten()(average_pooling)
    dense_1 = Dense(256, activation = 'relu')(flatten_)
    drop_3 = Dropout(0.25)(dense_1)
    dense_2 = Dense(128, activation = 'relu')(drop_3)
    soft_max = Dense(10, activation = 'softmax')(dense_2)
    # soft_max = Activation('softmax')

    model = Model(inputs = input_, outputs = soft_max)

    return model

def my_learning_rate(epoch, lrate):
    return lrate

model = Model_CNN()
print(model.summary())

# model = Model_VGG16()
lrs = LearningRateScheduler(my_learning_rate, verbose = 1)
sgd = SGD(lr = 0.01, decay = 0.01)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

X_train = X_train / 255
X_test = X_test / 255

# print(X_train[0].shape)
# X_train = np.expand_dims(X_train, axis = 3)
# X_test = np.expand_dims(X_test, axis = 3)

endp = int(len(X_train) * 0.9)

# X_val = X_train[endp:]
# y_val = y_train[endp:]

# X_train = X_train[:endp]
# y_train = y_train[:endp]
# X_train = np.moveaxis(X_train, 0, -1)

print(X_train.shape)

tsb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, 
                write_grads=False, write_images=False, embeddings_freq=0, 
                embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, 
                update_freq='epoch')

model.fit(X_train, y_train, epochs = 50)

loss, score = model.evaluate(X_test, y_test)

print(loss, score)


