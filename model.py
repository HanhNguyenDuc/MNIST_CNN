import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.callbacks import *




def Model_CNN():
    input_ = Input(shape = (28, 28, 1))
    conv_1a = Conv2D(8, kernel_size = (3, 3), padding = 'valid', activation = 'relu')(input_)
    conv_1b = Conv2D(8, kernel_size = (3, 3), padding = 'valid', activation = 'relu')(conv_1a)
    maxpool_1 = MaxPooling2D(pool_size = (2, 2), padding = 'valid')(conv_1b)   
    
    conv_2a = Conv2D(16, kernel_size = (3, 3), padding = 'valid', activation = 'relu')(maxpool_1)
    conv_2b = Conv2D(16, kernel_size = (3, 3), padding = 'valid', activation = 'relu')(conv_2a)
    maxpool_2 = MaxPooling2D(pool_size = (2, 2), padding = 'valid')(conv_2b)

    flatten_ = Flatten()(maxpool_2)
    dense_1 = Dense(128)(flatten_)
    activation_1 = Activation('relu')(dense_1)
    dense_2 = Dense(128)(activation_1)
    activation_2 = Activation('relu')(dense_2)
    soft_max = Dense(10, activation = 'softmax')(activation_2)
    # soft_max = Activation('softmax')

    model = Model(inputs = input_, outputs = soft_max)

    return model

def my_learning_rate(epoch, lrate):
    return lrate * (1 / (1 + 1e-3 * epoch))

model = Model_CNN()
print(model.summary())
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# model = Model_VGG16()
lrs = LearningRateScheduler(my_learning_rate, verbose = 1)
sgd = SGD(lr = 0.01, decay = 0.01)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

X_train = X_train / 255
X_test = X_test / 255

# print(X_train[0].shape)
X_train = np.expand_dims(X_train, axis = 3)
X_test = np.expand_dims(X_test, axis = 3)

endp = int(len(X_train) * 0.8)

X_val = X_train[endp:]
y_val = y_train[endp:]

X_train = X_train[:endp]
y_train = y_train[:endp]
# X_train = np.moveaxis(X_train, 0, -1)

print(X_train.shape)

tsb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, 
                write_grads=False, write_images=False, embeddings_freq=0, 
                embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, 
                update_freq='epoch')

model.fit(X_train, y_train, epochs = 5, callbacks = [lrs, tsb], validation_data = (X_val, y_val))

loss, score = model.evaluate(X_test, y_test)

print(loss, score)


