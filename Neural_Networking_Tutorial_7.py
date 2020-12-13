import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
import numpy as np


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model = Sequential()

model.add(LSTM(128, input_shape = (x_train.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))
model.add(Dropout(0.2))

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = opt,
    metrics = ['accuracy']
)

model.fit(x_train, y_train, epochs = 3, validation_data = (x_test, y_test))

model.save('/home/sherry/Code/Neural Network/Sentdex Tutorial/Hand_Written_Num_Reader_RNN/num_reader.model')
new_model = tf.keras.models.load_model('/home/sherry/Code/Neural Network/Sentdex Tutorial/Hand_Written_Num_Reader_RNN/num_reader.model')
predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))