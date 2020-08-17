import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import time
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras import backend as K

#Learning rate range test

def learning_rate_range_test(model, x_train, y_train, start_lr, end_lr, num_batches, batch_size, epochs):
    initial_weights = model.get_weights()

    #AdamW
    wd_schedule = tf.optimizers.schedules.PolynomialDecay(1e-4, 10000, 1e-5)
    adamw = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
    opt = adamw(weight_decay=lambda: None)
    opt.weight_decay = wd_schedule(opt.iterations)

    model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=opt,
    metrics=['accuracy']
)
    og_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.lr, start_lr)
    cb = LR_Finder(model, start_lr=start_lr, end_lr=end_lr, num_batches=num_batches)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[cb]
    )

    model.set_weights(initial_weights)
    K.set_value(model.optimizer.learning_rate, og_lr)
    return cb.lrs, cb.losses

class LR_Finder(Callback):
    def __init__(self, model, start_lr, end_lr, num_batches):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9
        self.num_batches = num_batches
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.initial_weights = model.get_weights()

    def on_batch_end(self, batch, logs):
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.lrs.append(lr)

        loss = logs['loss']
        self.losses.append(loss)

        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        self.lr_mult = (float(self.end_lr) / float(self.start_lr)) ** (float(1) / float(self.num_batches))

        lr *= self.lr_mult
        K.set_value(self.model.optimizer.learning_rate, lr)
        #self.model.set_weights(self.initial_weights)

#Data import and preprocessing

data_train = pd.read_csv('fashion-mnist_train.csv')
data_test = pd.read_csv('fashion-mnist_test.csv')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=13)

x_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train /= 255
x_test /= 255
x_val /= 255

#Model creation and training

batch_size = 256
num_classes = 10
epochs = 40
range_test_epochs = 6
num_batches = range_test_epochs * (len(x_train) / batch_size)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D(),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.summary()

lrs, losses = learning_rate_range_test(model, x_train, y_train, 1e-3, 0.1, num_batches, batch_size, range_test_epochs)
min_loss_idx = losses.index(min(losses))
print('Learning rate with lowest loss: ', lrs[min_loss_idx])

#Cyclical learning rate
lr_schedule = tfa.optimizers.Triangular2CyclicalLearningRate(float(lrs[min_loss_idx] * 0.25), lrs[min_loss_idx], 8 * epochs)

#AdamW
wd_schedule = tf.optimizers.schedules.PolynomialDecay(1e-4, 10000, 1e-5)
adamw = tfa.optimizers.extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
opt = adamw(weight_decay=lambda: None, learning_rate=lr_schedule)
opt.weight_decay = wd_schedule(opt.iterations)

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=opt,
    metrics=['accuracy']
)

mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(patience=4, verbose=1)

start_timer = time.time()

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=[mc, es]
)

stop_timer = time.time()

training_time = stop_timer - start_timer

print('Model trained for {} seconds'.format(training_time))

score = model.evaluate(x_test, y_test, verbose=1)