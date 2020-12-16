import os
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.layers import LeakyReLU
import keras.optimizers as optimizers

from sklearn.model_selection import train_test_split

import numpy as np

import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-b','--batch_size', type=int, default=128)
  parser.add_argument('-e','--epochs', type=int, default=16)
  parser.add_argument('-u','--units', type=int, default=24)
  parser.add_argument('-s','--save_dir', default='models')
  return parser.parse_args()

def rnn_model(num_classes, args):
  model = Sequential()
  model.add(SimpleRNN(args.units,
                      activation='tanh',
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='orthogonal',
                      input_shape=(28,28)
                      ))
  model.add(LeakyReLU(alpha=0.01))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
                metrics=['accuracy'])

  return model

if __name__ == '__main__':
  args = get_args()

  batch_size = args.batch_size
  epochs = args.epochs
  units = args.units

  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = (x_train.reshape(-1, 28, 28) / 255).astype(np.float32)
  x_test = (x_test.reshape(-1, 28, 28) / 255).astype(np.float32)

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

  os.makedirs(args.save_dir, exist_ok=True)
  model_checkpoint = ModelCheckpoint(
      filepath=os.path.join(args.save_dir, f'model_{units:02d}'+'_{epoch:02d}_{val_loss:.3f}.h5'),
      monitor='val_loss',
      verbose=1)

  model = rnn_model(num_classes, args)

  early_stop = EarlyStopping(monitor='val_loss',
                     patience=5,
                     verbose=1)

  history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_val, y_val), callbacks=[model_checkpoint, early_stop])

  scores = model.evaluate(x_test, y_test, verbose=0)

  print('loss:', scores[0])
  print('accuracy:', scores[1])
